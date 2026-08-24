module;
#include <array>
#include <complex>
#include <cstddef>
#include <span>
#include <utility>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <volk.h>
export module ses.scenario.base_director;
export import ses.grid;
export import ses.vk.engine_blobs;
export import ses.scenario;
export import ses.simulation;
export import ses.sampling;
export import ses.imaginary_time;
export import ses.observables;
export import ses.marching_cubes;
export import ses.field;
export import ses.potential;
export import ses.colormap;
export import ses.scenario.atom_model;
import ses.measurement;
import ses.photon_display;


// Shared machinery for potential-swap scenarios (trap, tunneling).
// HydrogenDirector reuses the members but overrides the whole frame flow.
// volk.h textually first: VK_* macros never cross module boundaries.


export namespace ses_shell {

enum class BaseViewMode { Cloud, Surface };
// RelaxingExcited: only HydrogenDirector (deflated relax) sets it.
enum class BaseStepping { RealTime, Relaxing, RelaxingExcited };

constexpr int kBaseStepsPerTick = 1;
constexpr int kBaseRelaxStepsPerTick = 1;
constexpr double kBaseRelaxDtau = 0.05;
constexpr double kBaseIsoFraction = 0.25;
constexpr double kBaseMeasureSigma = 1.25;  // Bohr
// Full-field readback ~10 ms: probe_readback fires every Nth title tick (~0.5 s).
constexpr int kBaseProbeStride = 3;
// ITP auto-complete plateau: energy step below eps for N title-cadence polls.
constexpr double kBaseRelaxPlateauEps = 5e-5;  // Ha
constexpr int kBaseRelaxPlateauPolls = 12;     // ~2 s of stable readout
// Collective-decay contract, ONE copy for hydrogen + trap (contract:
// tests/eigenstate_flush_test.cpp).
constexpr double kBaseGammaDisplay = 0.125;  // display decay rate
constexpr int kBaseFlushSteps = 6;
constexpr int kBaseFlushStepsGround = 24;  // the ground is the ITP fixed point
constexpr int kBaseFlashTicks = 25;  // photon-flash duration AND fade divisor
// Amplitude cutoff shared by superpose_into_psi's skip and callers' pre-filters.
constexpr double kAmpNormFloor = 1e-18;

// Shared photon-streak display: flight pool + per-slot overlay polylines.
// Directors spawn on a jump; run_frame advances and rebuilds (overlay
// pointers stay valid until the next run_frame).
struct PhotonStreakDisplay {
    ses::PhotonFlightPool pool;
    std::array<std::vector<float>, ses::kMaxPhotonFlights> xyz{};
    std::array<float, ses::kMaxPhotonFlights> alpha{};
    int count = 0;

    void spawn(const ses::PhotonRecord& rec, double delta_e) {
        pool.spawn(rec, delta_e, ses::photon_flight_frames(delta_e));
    }
    void advance_and_build() {
        pool.advance();
        count = pool.count();
        for (int c = 0; c < count; ++c) {
            const ses::PhotonFlightPool::Flight* f = pool.active(c);
            const double progress =
                static_cast<double>(f->age_frames) / f->total_frames;
            const std::vector<ses::Vec3d> pts = ses::photon_streak_vertices(
                f->rec, f->delta_e, progress);
            std::vector<float>& buf = xyz[static_cast<std::size_t>(c)];
            buf.resize(pts.size() * 3);
            for (std::size_t i = 0; i < pts.size(); ++i) {
                buf[3 * i + 0] = static_cast<float>(pts[i].x);
                buf[3 * i + 1] = static_cast<float>(pts[i].y);
                buf[3 * i + 2] = static_cast<float>(pts[i].z);
            }
            alpha[static_cast<std::size_t>(c)] =
                static_cast<float>(ses::photon_streak_alpha(progress));
        }
    }
    // LINE_STRIP helix, twist/rotation sense = the recorded helicity; warm
    // tint matches the flash language.
    OverlayCurve curve(int i) const {
        if (i < 0 || i >= count) {
            return {};
        }
        const std::vector<float>& buf = xyz[static_cast<std::size_t>(i)];
        OverlayCurve oc;
        oc.xyz = buf.data();
        oc.count = static_cast<int>(buf.size() / 3);
        oc.r = 1.0f;
        oc.g = 0.95f;
        oc.b = 0.75f;
        oc.a = alpha[static_cast<std::size_t>(i)];
        return oc;
    }
};

class BaseDirector : public ScenarioDirector {
public:
    explicit BaseDirector(ses::WavepacketSimulation sim) : sim_(std::move(sim)) {
        remesh();
        stage_volume();
    }

    const ses::Grid3D& grid() const override { return sim_.grid(); }

    // Engine/gradient failure demotes to CPU; absorber failure only drops the mask.
    void init_compute(ses_vk::DeviceContext& ctx, bool device_ok,
                      std::int64_t /*free_vram_bytes*/) override {
        compute_attempted_ = true;
        gpu_ok_ = device_ok &&
                  engine_.initialize(ctx, sim_.grid(),
                                     ses_vk::engine_blobs(sim_.grid().x.n),
                                     sim_.potential(), sim_.dt(),
                                     sim_.psi().data());
        if (!gpu_ok_) {
            return;
        }
        // Relax tables are transient (uploaded on relax entry); only gradient is fatal here.
        if (!engine_.set_potential_gradient(sim_.potential())) {
            std::fprintf(stderr, "engine: gradient setup failed -- "
                                 "falling back to CPU stepping\n");
            gpu_ok_ = false;
            return;
        }
        if (absorber_width() > 0.0) {
            absorber_on_ = engine_.set_absorber(
                ses::absorbing_mask(sim_.grid(), absorber_width()));
        }
        on_gpu_ready();
    }

    void release_gpu() override {
        engine_.destroy();
        gpu_ok_ = false;
    }

    bool use_gpu_path() const { return gpu_ok_; }

    void run_frame() override {
        // Photon streaks fly in wall-frame time, GPU path or not.
        photon_streaks_.advance_and_build();
        if (!use_gpu_path()) {
            return;
        }
        // Reclaim last frame's async batch FIRST (display flip + cb reuse).
        engine_.wait_async();
        bridge_cpu_state();
        service_requests();
        run_pending_batches();
        extract_surface_if_dirty();
    }

    void tick() override {
        if (use_gpu_path()) {
            // ONE tick's supply per frame (catch-up drops); time_scale_ is the
            // ONLY pacing dial -- no mode may add a hidden multiplier.
            const int per_tick = steps_per_tick() * time_scale_;
            pending_gpu_steps_ =
                pending_after_tick(pending_gpu_steps_, per_tick);
            if (++ticks_ % 10 == 0) {
                gpu_title_due_ = true;
            }
            return;
        }
        ensure_cpu_current();
        // CPU fallback: not time-scaled (sync steps would stall the UI).
        if (stepping_ == BaseStepping::RealTime) {
            sim_.advance(kBaseStepsPerTick);
        } else {
            sim_.relax(kBaseRelaxStepsPerTick, kBaseRelaxDtau);
        }
        stage_active_view();
        if (++ticks_ % 10 == 0) {
            norm_display_ = ses::norm_sq(sim_.psi());
            title_dirty_ = true;
        }
    }

    // steps per tick, not dt.
    void set_time_scale(int scale) override {
        time_scale_ = clamp_time_scale(scale);
    }
    int time_scale() const override { return time_scale_; }

    double sim_time() const override { return sim_.time() + gpu_time_; }
    double sim_dt() const override { return sim_.dt(); }

    // ---- generic controls ----

    void do_set_real_time() override {
        stepping_ = BaseStepping::RealTime;
        engine_.release_relax_tables();
    }

    void reset_simulation() override {
        sim_ = remake_simulation();
        stepping_ = BaseStepping::RealTime;
        cpu_is_truth_ = true;
        gpu_time_ = 0.0;
        pending_gpu_steps_ = 0;
        probe_phase_ = 0;
        after_reset();
        stage_active_view();
    }

    void measure_now() override {
        ensure_cpu_current();
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        sim_.measure(uniform(rng_), kBaseMeasureSigma);
        stepping_ = BaseStepping::RealTime;
        stage_active_view();
    }

    void toggle_view_mode() override {
        mode_ = (mode_ == BaseViewMode::Cloud) ? BaseViewMode::Surface
                                               : BaseViewMode::Cloud;
        if (mode_ == BaseViewMode::Surface) {
            if (gpu_ok_ && engine_.mc_prepare(kMcMaxTris)) {
                mc_dirty_ = true;
            } else {
                ensure_cpu_current();
            }
        } else {
            engine_.release_mc();
            if (gpu_ok_ && !cpu_is_truth_) {
                write_display_texture();
            }
        }
        stage_active_view();
    }

    void set_relaxing() {
        if (!relax_allowed()) {
            return;
        }
        if (use_gpu_path() && !ensure_relax_tables()) {
            return;
        }
        stepping_ = BaseStepping::Relaxing;
        relax_plateau_ = 0;
        relax_prev_energy_ = 0.0;
        if (!use_gpu_path()) {
            ensure_cpu_current();
        }
    }

    bool handle_key(char key) override {
        if (key == '2') {
            set_relaxing();
            return true;
        }
        return false;
    }

    bool solving() const override { return false; }
    bool scene_ready() const override { return compute_attempted_; }

    // ---- display accessors ----

    bool cloud() const override { return mode_ == BaseViewMode::Cloud; }
    VkBuffer surface_vbuf() const override {
        return (mode_ == BaseViewMode::Surface && engine_.mc_ready())
                   ? engine_.mc_vertex_buffer()
                   : VK_NULL_HANDLE;
    }
    VkBuffer surface_indirect() const override {
        return (mode_ == BaseViewMode::Surface && engine_.mc_ready())
                   ? engine_.mc_indirect_buffer()
                   : VK_NULL_HANDLE;
    }
    double peak() const override { return peak_; }
    bool compute_attempted() const override { return compute_attempted_; }
    bool gpu_ok() const override { return gpu_ok_; }
    VkImageView psi_volume_view() override {
        return gpu_ok_ ? engine_.volume_view() : VK_NULL_HANDLE;
    }
    VkImageView flow_velocity_view() override {
        return gpu_ok_ ? engine_.flow_velocity_view() : VK_NULL_HANDLE;
    }
    // Photon flash: a brief warm background right after a quantum jump
    // (flash_ticks_ stays 0 in scenes without jumps).
    float next_flash_intensity() override {
        if (flash_ticks_ <= 0) {
            return 0.0f;
        }
        const float w = static_cast<float>(
            ses::flash_intensity(flash_ticks_, kBaseFlashTicks));
        --flash_ticks_;
        return w;
    }
    long long photon_count() const override { return photon_count_; }

    // Photon streak overlays (count 0 until a scene spawns one).
    int overlay_curve_count() const override { return photon_streaks_.count; }
    OverlayCurve overlay_curve(int i) const override {
        return photon_streaks_.curve(i);
    }
    bool take_volume_written() override { return take(volume_written_); }
    bool take_volume_dirty() override { return take(volume_dirty_); }
    bool take_mesh_dirty() override { return take(mesh_dirty_); }
    bool take_title_dirty() override { return take(title_dirty_); }
    void mark_display_dirty() override {
        mesh_dirty_ = true;
        volume_dirty_ = true;
    }
    const std::vector<float>& psi_staging() const override { return psi_staging_; }
    const ses::Mesh& mesh() const override { return mesh_; }
    const std::vector<ses::Rgb>& colors() const override { return colors_; }

    std::string title_text() override {
        const double t_au = sim_.time() + gpu_time_;
        std::string s = scene_name() + std::string("   t = ") +
                        strf("{:.2f} au ({:.3f} fs)", t_au, t_au * kAuToFs) + "   ";
        if (stepping_ != BaseStepping::RealTime) {
            s += cpu_is_truth_
                     ? strf("E = {:.3f} eV   ",
                            ses::mean_energy(sim_.psi(), sim_.potential()) *
                                kHaToEv)
                     : strf("E ~ {:.3f} eV   ", relax_energy_display_ * kHaToEv);
        }
        if (stepping_ == BaseStepping::RealTime && use_gpu_path()) {
            s += strf("emit P = {:.2e} au   ", radiated_power_);
        }
        s += strf("norm = {:.6f}   [{}, {}, {}]  1=real 2=relax R=reset tab=view "
                  "[ ]=density M=pos",
                  norm_display_,
                  mode_ == BaseViewMode::Cloud ? "cloud" : "surface",
                  stepping_ == BaseStepping::RealTime ? "real-time"
                                                      : "relaxing->ground",
                  use_gpu_path() ? "gpu 256^3" : "cpu 256^3");
        s += title_suffix();
        return s;
    }

protected:
    // ---- scenario hooks ----
    virtual ses::WavepacketSimulation remake_simulation() const = 0;
    virtual const char* scene_name() const = 0;
    virtual std::string title_suffix() { return std::string(); }
    virtual double absorber_width() const { return 0.0; }
    virtual void on_gpu_ready() {}
    virtual void after_reset() {}
    virtual void service_requests() {}   // run_frame, before stepping
    virtual void after_step_batch() {}   // run_frame, after a real-time batch
    virtual int steps_per_tick() const { return kBaseStepsPerTick; }
    virtual bool relax_allowed() const { return true; }

    static bool take(bool& flag) { return std::exchange(flag, false); }

    // CPU state authoritative: refresh the brightness normalizer, upload,
    // bridge immediately (an empty step queue would keep a stale cloud).
    void bridge_cpu_state() {
        if (!cpu_is_truth_) {
            return;
        }
        double pk = 0.0;
        for (const std::complex<double>& z : sim_.psi().data()) {
            pk = std::max(pk, std::norm(z));
        }
        if (pk > 0.0) {
            peak_ = pk;
        }
        engine_.upload_state(sim_.psi().data());
        cpu_is_truth_ = false;
        volume_dirty_ = false;  // texture comes from the bridge now
        write_display_texture();
    }

    // Run this frame's queued steps (real-time or relax batch) + the title
    // epilogue.
    void run_pending_batches() {
        if (pending_gpu_steps_ <= 0) {
            return;
        }
        if (stepping_ == BaseStepping::RealTime) {
            run_real_time_batch();
            if (mode_ == BaseViewMode::Cloud) {
                volume_written_ = true;
            } else {
                mc_dirty_ = true;  // psi advanced: re-extract
            }
        } else {
            run_relax_batch();
            write_display_texture();
        }
        pending_gpu_steps_ = 0;
        volume_dirty_ = false;
        if (gpu_title_due_) {
            gpu_title_due_ = false;
            title_dirty_ = true;
        }
    }

    // kBaseIsoFraction * peak mirrors marching_cubes_at_fraction (CPU/GPU iso
    // parity).
    void extract_surface_if_dirty() {
        if (mode_ == BaseViewMode::Surface && engine_.mc_ready() &&
            mc_dirty_) {
            engine_.mc_extract(kBaseIsoFraction * peak_);
            mc_dirty_ = false;
            volume_written_ = true;  // display changed: accumulation resets
        }
    }

    // Virtual: HydrogenDirector substitutes the driven/magnetic/decay batch.
    virtual void run_real_time_batch() {
        if (gpu_title_due_) {
            const ses_vk::Engine::NormPeak np = engine_.norm_and_peak();
            norm_display_ = np.sum;
            if (np.peak > 0.0) {
                peak_ = np.peak;
            }
            // fp32 drift renormalization (split-operator is unitary exactly).
            if (np.sum > 0.0 && std::abs(np.sum - 1.0) > 1e-4 &&
                absorber_width() == 0.0) {
                engine_.scale(static_cast<float>(1.0 / std::sqrt(np.sum)));
            }
            radiated_power_ = ses::larmor_power(engine_.mean_force());
        }
        // ASYNC: overlaps this frame's render; next run_frame waits and flips.
        // after_step_batch hooks reading psi serialize on the same queue.
        engine_.step_async(pending_gpu_steps_,
                           {.absorb = absorber_on_, .bridge = true});
        gpu_time_ += pending_gpu_steps_ * sim_.dt();
        after_step_batch();
    }

    // Virtual: molecule scenes substitute a deflated-excited batch, reusing the rest.
    virtual void run_relax_batch() {
        const ses_vk::Engine::RelaxStats stats =
            engine_.relax_step(pending_gpu_steps_);
        relax_energy_display_ = stats.energy;
        if (stats.peak > 0.0) {
            peak_ = stats.peak;
        }
        norm_display_ = 1.0;  // pinned by per-step renormalization
        // Auto-complete on the ITP energy plateau.
        if (relax_plateau_poll(stats.energy)) {
            stepping_ = BaseStepping::RealTime;
            engine_.release_relax_tables();
        }
    }

    // ITP plateau poll (title cadence): true once the energy step stayed under
    // eps for kBaseRelaxPlateauPolls polls; the counter resets on completion.
    // The caller epilogues (stepping flip, table release, scene cleanup).
    bool relax_plateau_poll(double energy) {
        if (!gpu_title_due_) {
            return false;
        }
        if (std::abs(energy - relax_prev_energy_) < kBaseRelaxPlateauEps) {
            ++relax_plateau_;
        } else {
            relax_plateau_ = 0;
        }
        relax_prev_energy_ = energy;
        if (relax_plateau_ < kBaseRelaxPlateauPolls) {
            return false;
        }
        relax_plateau_ = 0;
        return true;
    }

    // Subtract every tracked amplitude out of psi (continuum / untracked
    // verdict): a bound population must not survive it. No-op under the fp32
    // noise floor.
    void project_manifold_out(AtomModel& atom) {
        std::vector<std::complex<double>> amp(
            static_cast<std::size_t>(atom.n_states()));
        double bound = 0.0;
        for (int s = 0; s < atom.n_states(); ++s) {
            amp[static_cast<std::size_t>(s)] =
                atom.project_state_amplitude(engine_, s);
            bound += std::norm(amp[static_cast<std::size_t>(s)]);
        }
        const double residual = engine_.norm_and_peak().sum - bound;
        if (residual <= 1e-4) {
            return;  // fp32 noise floor
        }
        for (int s = 0; s < atom.n_states(); ++s) {
            const std::complex<double> c = amp[static_cast<std::size_t>(s)];
            if (std::norm(c) < 1e-9) {
                continue;
            }
            const TransientState buf(atom, engine_, s);
            if (buf) {
                engine_.add_state_into_psi(buf.get(), -c.real(), -c.imag());
            }
        }
        const ses_vk::Engine::NormPeak np = engine_.norm_and_peak();
        if (np.sum > 1e-12) {
            engine_.scale(static_cast<float>(1.0 / std::sqrt(np.sum)));
        }
        cpu_is_truth_ = false;
        write_display_texture();
    }

    // Born-sample an energy eigenstate from the projected populations and
    // collapse onto it; -1 = outside the tracked manifold (projected OUT).
    // Caller flushes / labels / tallies.
    int measure_energy_collapse(AtomModel& atom) {
        engine_.project_psi();
        std::vector<double> pop(static_cast<std::size_t>(atom.n_states()));
        for (int s = 0; s < atom.n_states(); ++s) {
            pop[static_cast<std::size_t>(s)] =
                atom.project_population(engine_, s);
        }
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const int n = ses::sample_energy_eigenstate(pop, uniform(rng_));
        if (n >= 0) {
            atom.collapse_onto(engine_, n);
        } else {
            project_manifold_out(atom);
        }
        return n;
    }

    // ---- collective decay (hydrogen + trap share ONE trial pipeline) ----

    // Scene hooks around run_collective_decay_trial / flush_collapse_error.
    virtual void on_decay_jump(const ses::IntervalJump&) {}  // label/tally/log
    virtual void before_collapse_flush() {}
    virtual bool collapse_flush_allowed() const { return true; }
    virtual int ground_flush_index() const { return 0; }
    virtual void after_collapse_flush(int /*target*/, double /*energy*/) {}

    // Post-collapse eigenstate-error flush (fixed budget): collapse targets
    // are SAMPLED radial eigenstates, not grid eigenstates, so a short ITP
    // burst flushes high-frequency junk before real time resumes. Tables stay
    // resident while decay is armed (rebuilding per photon costs more than
    // the burst); a one-off flush drops them again.
    void flush_collapse_error(int target) {
        if (!collapse_flush_allowed() || !ensure_relax_tables()) {
            return;
        }
        const ses_vk::Engine::RelaxStats stats = engine_.relax_step(
            target == ground_flush_index() ? kBaseFlushStepsGround
                                           : kBaseFlushSteps);
        if (!decay_on_) {
            engine_.release_relax_tables();
        }
        after_collapse_flush(target, stats.energy);
    }

    struct DecayTrialResult {
        bool jumped = false;
        double trial_dt = 0.0;  // the interval the trial consumed
    };

    // One frequency-resolved collective trial on the accumulated interval:
    // project tracked amplitudes -> ses::group_by_gap ->
    // ses::collective_decay_interval -> rebuild psi (superpose; collapse_onto
    // guard fallback) -> flush -> flash -> one streak + on_decay_jump per
    // arrival. PRECONDITION: project_psi() ran on the CURRENT psi.
    // first_state: projection loop start (hydrogen skips the stable ground).
    // pop_out (n_states doubles, optional): |amp|^2 tap for the MCWF no-jump
    // branch. RNG CONTRACT: every draw goes through ONE uniform dist fed to
    // the sampler lambda -- same seed, same jump sequence.
    DecayTrialResult run_collective_decay_trial(AtomModel& atom,
                                                int first_state,
                                                double* pop_out = nullptr) {
        const int n = atom.n_states();
        std::vector<std::complex<double>> amps(static_cast<std::size_t>(n));
        for (int s = first_state; s < n; ++s) {
            amps[static_cast<std::size_t>(s)] =
                atom.project_state_amplitude(engine_, s);
            if (pop_out != nullptr) {
                pop_out[static_cast<std::size_t>(s)] =
                    std::norm(amps[static_cast<std::size_t>(s)]);
            }
        }
        const std::vector<ses::FreqGroup> groups =
            ses::group_by_gap(atom.grouped_channels(), ses::kFreqGroupTol);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const double trial_dt = decay_accum_dt_;
        decay_accum_dt_ = 0.0;
        const ses::IntervalResult res = ses::collective_decay_interval(
            groups, std::move(amps), trial_dt, [&] { return uniform(rng_); });
        if (res.jumps.empty()) {
            return {.jumped = false, .trial_dt = trial_dt};
        }
        std::vector<int> states;
        std::vector<std::complex<double>> cs;
        for (int s = 0; s < n; ++s) {
            if (std::norm(res.c[static_cast<std::size_t>(s)]) >
                kAmpNormFloor) {
                states.push_back(s);
                cs.push_back(res.c[static_cast<std::size_t>(s)]);
            }
        }
        const int dom = res.jumps.back().dominant_to;
        if (!superpose_into_psi(atom, states, cs) && dom >= 0) {
            atom.collapse_onto(engine_, dom);  // guard fallback
        }
        before_collapse_flush();
        flush_collapse_error(dom >= 0 ? dom : ground_flush_index());
        flash_ticks_ = kBaseFlashTicks;
        for (const ses::IntervalJump& j : res.jumps) {
            ++photon_count_;
            photon_streaks_.spawn(j.rec, j.gap_e);
            on_decay_jump(j);
        }
        title_dirty_ = true;
        return {.jumped = true, .trial_dt = trial_dt};
    }

    // THE synth-accumulate idiom (seeds, partial-measure rebuild, shell
    // collapse): overwrite psi with sum_i c_i |states_i>. First significant
    // member anchors with its phase rotated out (global phase unphysical; the
    // real-only engine scale stays exact); near-zero and failed-synth members
    // are skipped. Renormalizes, refreshes peak_/norm_display_, invalidates
    // cpu_is_truth_. False = nothing anchored (psi untouched).
    bool superpose_into_psi(AtomModel& atom, std::span<const int> states,
                            std::span<const std::complex<double>> c) {
        bool anchored = false;
        std::complex<double> phase{1.0, 0.0};
        for (std::size_t i = 0; i < states.size(); ++i) {
            if (std::norm(c[i]) < kAmpNormFloor) {
                continue;
            }
            const TransientState buf(atom, engine_, states[i]);
            if (!buf) {
                continue;
            }
            if (!anchored) {
                phase = std::conj(c[i]) / std::abs(c[i]);
                engine_.copy_into_psi(buf.get());
                engine_.scale(static_cast<float>(std::abs(c[i])));
                anchored = true;
            } else {
                const std::complex<double> z = c[i] * phase;
                engine_.add_state_into_psi(buf.get(), z.real(), z.imag());
            }
        }
        if (!anchored) {
            return false;
        }
        const ses_vk::Engine::NormPeak np = engine_.norm_and_peak();
        if (np.sum > 0.0) {
            engine_.scale(static_cast<float>(1.0 / std::sqrt(np.sum)));
            peak_ = np.peak / np.sum;
        }
        norm_display_ = 1.0;
        cpu_is_truth_ = false;
        return true;
    }

    // Relax/ITP tables ride THIS potential; static-field scenes override to
    // the field-dressed effective V (else imaginary time cools to the
    // field-free ground).
    virtual const std::vector<double>& relax_potential() const {
        return sim_.potential();
    }

    bool ensure_relax_tables() {
        if (engine_.relax_tables_ready()) {
            return true;
        }
        const double dtau = relax_dtau();
        const ses::ImaginaryTimePropagator3D relaxer{sim_.grid(),
                                                     relax_potential(), dtau};
        if (!engine_.set_relax_tables(relaxer.half_potential_weight(),
                                      relaxer.kinetic_weight(), dtau,
                                      sim_.grid().cell_volume())) {
            std::fprintf(stderr, "engine: relax table upload failed\n");
            return false;
        }
        return true;
    }

    // Deep wells override: V*dtau must stay moderate or the ITP fixed point is a
    // Trotter artifact, not the grid-H eigenstate.
    virtual double relax_dtau() const { return kBaseRelaxDtau; }

    void ensure_cpu_current() {
        pending_gpu_steps_ = 0;  // uncredited steps must not fire later
        if (cpu_is_truth_ || !gpu_ok_) {
            return;
        }
        if (!engine_.readback(readback_buf_)) {
            std::fprintf(stderr,
                         "engine: readback failed -- keeping the CPU state\n");
            return;
        }
        sim_.set_psi(field_from_readback());
        cpu_is_truth_ = true;
    }

    // Gated density probe over the full field (kBaseProbeStride title-tick
    // cadence + host-wait + readback-failure skip in ONE place). cell gets
    // (i, j, k, |psi|^2) per grid cell, x-fastest layout. False = skipped.
    template <typename CellFn>
    bool probe_readback(CellFn&& cell) {
        if (!gpu_title_due_ || ++probe_phase_ % kBaseProbeStride != 0) {
            return false;
        }
        // Host-wait: readback consumes POST-step psi (same-queue order carries
        // no memory dependency).
        engine_.wait_async();
        if (!engine_.readback(readback_buf_)) {
            return false;  // GPU readback failed: skip (no stale/OOB read)
        }
        const ses::Grid3D& g = sim_.grid();
        const int nx = g.x.n;
        const int ny = g.y.n;
        const std::size_t cells = readback_buf_.size() / 2;
        for (std::size_t idx = 0; idx < cells; ++idx) {
            const double re = readback_buf_[2 * idx];
            const double im = readback_buf_[2 * idx + 1];
            cell(static_cast<int>(idx % nx),
                 static_cast<int>((idx / nx) % ny),
                 static_cast<int>(idx / (nx * ny)), re * re + im * im);
        }
        return true;
    }

    // readback_buf_ (interleaved fp32 re/im) -> Field3D on the sim grid.
    ses::Field3D field_from_readback() const {
        ses::Field3D f{sim_.grid()};
        for (std::size_t i = 0; i < f.data().size(); ++i) {
            f.data()[i] = std::complex<double>{readback_buf_[2 * i],
                                               readback_buf_[2 * i + 1]};
        }
        return f;
    }

    void stage_active_view() {
        if (mode_ == BaseViewMode::Cloud) {
            if (use_gpu_path()) {
                return;  // run_frame uploads and bridges
            }
            stage_volume();
            volume_dirty_ = true;
        } else {
            if (gpu_ok_) {
                mc_dirty_ = true;
                return;
            }
            remesh();
            mesh_dirty_ = true;
        }
    }

    void remesh() {
        mesh_ = ses::marching_cubes_at_fraction(sim_.density(), sim_.grid(),
                                                kBaseIsoFraction);
        colors_ = ses::phase_colors(mesh_, sim_.psi());
    }

    void stage_volume() {
        const auto& field = sim_.psi().data();
        psi_staging_.resize(field.size() * 2);
        double peak = 0.0;
        for (std::size_t i = 0; i < field.size(); ++i) {
            psi_staging_[2 * i] = static_cast<float>(field[i].real());
            psi_staging_[2 * i + 1] = static_cast<float>(field[i].imag());
            peak = std::max(peak, std::norm(field[i]));
        }
        peak_ = peak;
    }

    void write_display_texture() {
        if (mode_ == BaseViewMode::Surface && engine_.mc_ready()) {
            mc_dirty_ = true;
            return;
        }
        engine_.write_psi_to_volume();
        volume_written_ = true;  // resets the temporal accumulation
    }

    ses::WavepacketSimulation sim_;
    ses_vk::Engine engine_;
    BaseViewMode mode_ = BaseViewMode::Cloud;
    BaseStepping stepping_ = BaseStepping::RealTime;
    bool compute_attempted_ = false;
    bool gpu_ok_ = false;
    bool cpu_is_truth_ = true;
    int pending_gpu_steps_ = 0;
    int time_scale_ = 1;
    bool gpu_title_due_ = false;
    bool title_dirty_ = false;
    double gpu_time_ = 0.0;
    double norm_display_ = 1.0;
    double relax_energy_display_ = 0.0;
    double radiated_power_ = 0.0;  // au
    double relax_prev_energy_ = 0.0;
    int relax_plateau_ = 0;
    std::vector<float> readback_buf_;
    int probe_phase_ = 0;  // probe_readback cadence counter

    bool mc_dirty_ = false;  // Surface: re-extract mesh
    ses::Mesh mesh_;
    std::vector<ses::Rgb> colors_;
    std::vector<float> psi_staging_;
    double peak_ = 0.0;
    bool mesh_dirty_ = false;
    bool volume_dirty_ = false;
    bool volume_written_ = false;
    long long ticks_ = 0;

    bool absorber_on_ = false;
    std::mt19937 rng_{std::random_device{}()};
    PhotonStreakDisplay photon_streaks_;

    // Shared decay/flash/measure scene state (hydrogen + trap).
    bool decay_on_ = false;
    double decay_accum_dt_ = 0.0;  // sim time since the last decay trial
    int flash_ticks_ = 0;
    long long photon_count_ = 0;
    int excite_cycle_ = 0;  // excite-key cycle position
    std::string last_jump_;
    std::string last_measure_;
};

}  // namespace ses_shell
