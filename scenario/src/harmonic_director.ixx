module;
#include <numbers>
#include <algorithm>
#include <cstddef>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
export module ses.scenario.harmonic_director;
export import ses.scenario.base_director;
import ses.scenario.atom_model;
import ses.photon_display;
import ses.projection;
import ses.measurement;


// 3D isotropic harmonic trap. Central potential -> tracked-manifold applies
// (license: tests/trap_ladder_test.cpp).


export namespace ses_shell {

constexpr double kTrapOmega = 0.25;      // au
constexpr double kTrapBox = 20.0;        // Bohr half-extent
constexpr double kCoherentOffset = 8.0;  // Bohr

// Ladder N = 2k+l <= 3, E = (N + 3/2) w; state order = AtomModel index convention.
inline constexpr int kNumTrapLevels = 6;
inline constexpr RadialLevelSpec kTrapLevels[kNumTrapLevels] = {
    {0, 0}, {1, 0}, {0, 1}, {2, 0}, {1, 1}, {3, 0},
};
inline constexpr int kNumTrapStates = 20;
inline constexpr StateSpec kTrapStates[kNumTrapStates] = {
    {0, 0, 0, "0s"},
    {1, 1, 1, "1p_x"}, {1, 1, -1, "1p_y"}, {1, 1, 0, "1p_z"},
    {2, 0, 0, "2s"},
    {3, 2, -2, "2d_xy"}, {3, 2, -1, "2d_yz"}, {3, 2, 0, "2d_z2"},
    {3, 2, 1, "2d_zx"}, {3, 2, 2, "2d_x2y2"},
    {4, 1, 1, "3p_x"}, {4, 1, -1, "3p_y"}, {4, 1, 0, "3p_z"},
    {5, 3, -3, "3f_-3"}, {5, 3, -2, "3f_-2"}, {5, 3, -1, "3f_-1"},
    {5, 3, 0, "3f_0"}, {5, 3, 1, "3f_+1"}, {5, 3, 2, "3f_+2"},
    {5, 3, 3, "3f_+3"},
};

// Display decay + post-collapse flush budgets (contract: eigenstate_flush_test).
constexpr double kTrapGammaDisplay = 0.125;
constexpr int kTrapFlushSteps = 6;
constexpr int kTrapFlushStepsGround = 24;  // 0s is the ITP fixed point
constexpr int kTrapFlashTicks = 25;

class HarmonicDirector final : public BaseDirector {
public:
    HarmonicDirector() : BaseDirector(make()) {}

    bool handle_key(char key) override {
        if (BaseDirector::handle_key(key)) {
            return true;
        }
        switch (key) {
            case '5':
                excite_next();
                return true;
            case 'D':
                toggle_decay();
                return true;
            case 'E':
                measure_energy_now();
                return true;
            default:
                return false;
        }
    }

    float next_flash_intensity() override {
        if (flash_ticks_ <= 0) {
            return 0.0f;
        }
        const float w = static_cast<float>(
            ses::flash_intensity(flash_ticks_, kTrapFlashTicks));
        --flash_ticks_;
        return w;
    }

    long long photon_count() const override { return photon_count_; }

protected:
    ses::WavepacketSimulation remake_simulation() const override { return make(); }
    const char* scene_name() const override { return "Harmonic trap"; }
    double default_camera_distance() const override { return 45.0; }

    std::string title_suffix() override {
        std::string s = strf("  w = {:.2f} au (T = {:.1f} au, E0 = {:.2f} eV)",
                             kTrapOmega, (2.0 * std::numbers::pi) / kTrapOmega,
                             1.5 * kTrapOmega * kBaseHaToEv);
        if (decay_on_) {
            s += strf("  decay ON: photons {}", photon_count_);
            if (!last_jump_.empty()) {
                s += strf(", last {}", last_jump_.c_str());
            }
        }
        if (!last_measure_.empty()) {
            s += strf("  measured {}", last_measure_.c_str());
        }
        s += "  5=excite D=decay E=measure";
        return s;
    }

    // Key E deferred to run_frame: psi must be quiescent before stepping.
    void service_requests() override {
        if (!pending_measure_) {
            return;
        }
        pending_measure_ = false;
        const int n = measure_energy_collapse(atom_);
        if (n >= 0) {
            flush_collapse_error(n);
            write_display_texture();
            last_measure_ = strf("{} (E = {:.2f} eV)", kTrapStates[n].name,
                                 atom_.state_energy(n) * kBaseHaToEv);
        } else {
            // Deficit = untracked bound ladder (N > 3), not continuum.
            last_measure_ = "outside tracked ladder (N > 3)";
        }
        title_dirty_ = true;
    }

    // Decay trials at title cadence: COLLECTIVE jumps. Every trap gap is
    // exactly hbar*omega, so the photon cannot resolve which rung fired --
    // the jump operator is the collective lowering sum (damped-HO L ~ a per
    // axis, ses::collective_jump_dipoles); rung coherences survive and
    // arrivals chain analytically within the interval, one streak each.
    void after_step_batch() override {
        if (!decay_on_ || atom_.channels().empty()) {
            return;
        }
        decay_accum_dt_ += pending_gpu_steps_ * sim_.dt();
        if (!gpu_title_due_) {
            return;
        }
        engine_.wait_async();  // the deposit needs the batch's memory visible
        engine_.project_psi();
        std::vector<std::complex<double>> c(
            static_cast<std::size_t>(kNumTrapStates));
        for (int s = 0; s < kNumTrapStates; ++s) {
            c[static_cast<std::size_t>(s)] =
                atom_.project_state_amplitude(engine_, s);
        }
        const std::vector<ses::DipoleChannel> dch = atom_.dipole_channels();
        // Display-rate scale: gamma_ch = k_rate * m_ch^2 with ONE shared k
        // (equal gaps make it channel-independent).
        double k_rate = 0.0;
        for (std::size_t i = 0; i < dch.size(); ++i) {
            const double m2 = dch[i].mx * dch[i].mx + dch[i].my * dch[i].my +
                              dch[i].mz * dch[i].mz;
            if (m2 > 0.0) {
                k_rate = atom_.channels()[i].gamma_display / m2;
                break;
            }
        }
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        double remaining = decay_accum_dt_;
        decay_accum_dt_ = 0.0;
        int jumps = 0;
        int dom = 0;  // dominant destination of the LAST jump (flush target)
        std::vector<int> final_states;
        std::vector<std::complex<double>> final_c;
        for (;;) {
            const ses::CollectiveDipoles cd =
                ses::collective_jump_dipoles(dch, c);
            double wsum = 0.0;
            for (const ses::DipoleMatrixElement& d : cd.dipoles) {
                wsum += std::norm(d.x) + std::norm(d.y) + std::norm(d.z);
            }
            const double rate = k_rate * wsum;
            if (rate <= 0.0) {
                break;
            }
            const double t1 = -std::log(uniform(rng_)) / rate;
            if (!(t1 < remaining)) {
                break;
            }
            remaining -= t1;
            const ses::PhotonRecord rec = ses::sample_photon_emission(
                cd.dipoles, [&] { return uniform(rng_); });
            const std::vector<std::complex<double>> cond =
                ses::conditioned_amplitudes(cd.dipoles, rec.n, rec.helicity);
            std::fill(c.begin(), c.end(), std::complex<double>{});
            double best = 0.0;
            dom = cd.to_states[0];
            for (std::size_t i = 0; i < cd.to_states.size(); ++i) {
                c[static_cast<std::size_t>(cd.to_states[i])] = cond[i];
                if (std::norm(cond[i]) > best) {
                    best = std::norm(cond[i]);
                    dom = cd.to_states[i];
                }
            }
            final_states = cd.to_states;
            final_c = cond;
            photon_streaks_.spawn(rec, kTrapOmega);  // every photon = hbar*w
            ++jumps;
            ++photon_count_;
            last_jump_ = strf("hw -> {}", kTrapStates[dom].name);
            std::fprintf(
                stderr,
                "trap decay: collective jump -> %s (photon #%lld, t=%.1f au)\n",
                kTrapStates[dom].name, photon_count_, sim_.time() + gpu_time_);
        }
        if (jumps == 0) {
            return;
        }
        if (superpose_into_psi(atom_, final_states, final_c)) {
            flush_collapse_error(dom);
        } else {
            atom_.collapse_onto(engine_, dom);  // guard fallback
            flush_collapse_error(dom);
        }
        flash_ticks_ = kTrapFlashTicks;
        title_dirty_ = true;
    }

private:
    // Coherent state at ground width (no breathing); harmonic_dynamics_test pins kSigmaGs.
    static ses::WavepacketSimulation make() {
        const ses::Grid1D axis{-kTrapBox, kTrapBox, 256};
        const ses::Grid3D grid{axis, axis, axis};
        const double sigma = 1.0 / std::sqrt(2.0 * kTrapOmega);
        return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
            grid,
            ses::harmonic_potential(grid, kTrapOmega, ses::Vec3d{}),
            ses::Vec3d{kCoherentOffset, 0.0, 0.0},
            ses::Vec3d{sigma, sigma, sigma},
            ses::Vec3d{},  // initial velocity: at rest
            0.04,          // dt
        }};
    }

    // Lazy: built on first D/E only; plain scene never pays.
    bool ensure_manifold() {
        if (!use_gpu_path()) {
            return false;
        }
        if (!atom_.radial_ready()) {
            const ses::RadialGrid rg{kTrapBox, 3999};
            std::vector<double> vr(static_cast<std::size_t>(rg.n));
            for (int i = 0; i < rg.n; ++i) {
                const double r = rg.r(i);
                vr[static_cast<std::size_t>(i)] =
                    0.5 * kTrapOmega * kTrapOmega * r * r;
            }
            atom_.solve_radial_manifold(rg, vr, kTrapStates, kNumTrapStates,
                                        kTrapLevels, kNumTrapLevels);
        }
        if (!proj_ready_) {
            const ses::RadialBinIndex bin_idx =
                ses::build_radial_bin_index(sim_.grid(), atom_.radial_grid());
            proj_ready_ = engine_.set_projection_index(
                bin_idx.sorted_cell, bin_idx.bin_off, atom_.radial_grid().n,
                atom_.radial_grid().h(), 3);  // l_max = 3 (N <= 3 ladder)
            if (!proj_ready_) {
                std::fprintf(stderr, "trap: projection index setup failed -- "
                                     "measurement/decay disabled\n");
                return false;
            }
        }
        return atom_.prepare_manifold_cache(engine_, kTrapGammaDisplay);
    }

    void toggle_decay() {
        if (!decay_on_) {
            if (!ensure_manifold()) {
                return;
            }
            decay_accum_dt_ = 0.0;  // no hazard accrues while off
        }
        decay_on_ = !decay_on_;
        // Flush-table residency ends with decay (atom policy).
        if (!decay_on_ && stepping_ == BaseStepping::RealTime) {
            engine_.release_relax_tables();
        }
    }

    void measure_energy_now() {
        if (!ensure_manifold()) {
            return;
        }
        pending_measure_ = true;
        stepping_ = BaseStepping::RealTime;
    }

    void excite_next() {
        if (!ensure_manifold()) {
            return;
        }
        static constexpr int kCycle[] = {3, 7, 4, 12};  // 1p_z 2d_z2 2s 3p_z
        const int idx = kCycle[excite_cycle_++ % 4];
        atom_.collapse_onto(engine_, idx);
        flush_collapse_error(idx);
        cpu_is_truth_ = false;
        stepping_ = BaseStepping::RealTime;
        last_measure_.clear();
        title_dirty_ = true;
    }

    // Post-collapse eigenstate-error flush (fixed budget; mirrors atom policy).
    void flush_collapse_error(int target) {
        if (!ensure_relax_tables()) {
            return;
        }
        engine_.relax_step(target == 0 ? kTrapFlushStepsGround
                                       : kTrapFlushSteps);
        if (!decay_on_) {
            engine_.release_relax_tables();
        }
    }

    // Collapse onto untracked-ladder complement (N > 3).
    ses_shell::AtomModel atom_;
    bool proj_ready_ = false;
    bool decay_on_ = false;
    bool pending_measure_ = false;
    double decay_accum_dt_ = 0.0;
    long long photon_count_ = 0;
    int excite_cycle_ = 0;
    int flash_ticks_ = 0;
    std::string last_jump_;
    std::string last_measure_;
};

}  // namespace ses_shell
