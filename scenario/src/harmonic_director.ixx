module;
#include <numbers>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iterator>
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
// Key-5 excite cycle targets (kTrapStates order).
constexpr int kTrap1PZ = 3;
constexpr int kTrap2S = 4;
constexpr int kTrap2DZ2 = 7;
constexpr int kTrap3PZ = 12;

constexpr int kTrapRadialSamples = 3999;
// Projection-index l cap, derived from the ladder spec (N <= 3 -> f).
inline constexpr int trap_l_max() {
    int m = 0;
    for (const RadialLevelSpec& lev : kTrapLevels) {
        m = lev.l > m ? lev.l : m;
    }
    return m;
}
constexpr int kTrapLMax = trap_l_max();

// Decay/flush/flash contract: base kBaseGammaDisplay/kBaseFlush*/kBaseFlashTicks.

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

    // next_flash_intensity / photon_count: BaseDirector's.

protected:
    ses::WavepacketSimulation remake_simulation() const override { return make(); }
    const char* scene_name() const override { return "Harmonic trap"; }
    double default_camera_distance() const override { return 45.0; }

    std::string title_suffix() override {
        std::string s = strf("  w = {:.2f} au (T = {:.1f} au, E0 = {:.2f} eV)",
                             kTrapOmega, (2.0 * std::numbers::pi) / kTrapOmega,
                             1.5 * kTrapOmega * kHaToEv);
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
                                 atom_.state_energy(n) * kHaToEv);
        } else {
            // Deficit = untracked bound ladder (N > 3), not continuum.
            last_measure_ = "outside tracked ladder (N > 3)";
        }
        title_dirty_ = true;
    }

    // Decay trials at title cadence through the SAME frequency-group core as
    // hydrogen (base run_collective_decay_trial): every trap gap is
    // hbar*omega, so group_by_gap collapses to ONE group -- the damped-HO
    // collective limit (rung coherences survive; in-interval chains are
    // exact; one streak per arrival). Projects ALL states (0s decays into
    // nothing but superposes).
    void after_step_batch() override {
        if (!decay_on_ || atom_.channels().empty()) {
            return;
        }
        decay_accum_dt_ += pending_gpu_steps_ * sim_.dt();
        if (!gpu_title_due_) {
            return;
        }
        engine_.wait_async();  // the deposit needs the batch's memory visible
        assert(proj_ready_);   // trials read the project_psi deposit
        engine_.project_psi();
        run_collective_decay_trial(atom_, 0);
    }

    void on_decay_jump(const ses::IntervalJump& j) override {
        const char* name =
            kTrapStates[static_cast<std::size_t>(j.dominant_to)].name;
        last_jump_ = strf("hw -> {}", name);
        std::fprintf(
            stderr,
            "trap decay: collective jump -> %s (photon #%lld, t=%.1f au)\n",
            name, photon_count_, sim_.time() + gpu_time_);
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
            const ses::RadialGrid rg{kTrapBox, kTrapRadialSamples};
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
                atom_.radial_grid().h(), kTrapLMax);
            if (!proj_ready_) {
                std::fprintf(stderr, "trap: projection index setup failed -- "
                                     "measurement/decay disabled\n");
                return false;
            }
        }
        return atom_.prepare_manifold_cache(engine_, kBaseGammaDisplay);
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
        static constexpr int kCycle[] = {kTrap1PZ, kTrap2DZ2, kTrap2S,
                                         kTrap3PZ};
        const int idx = kCycle[static_cast<std::size_t>(excite_cycle_) %
                               std::size(kCycle)];
        ++excite_cycle_;
        atom_.collapse_onto(engine_, idx);
        flush_collapse_error(idx);
        cpu_is_truth_ = false;
        stepping_ = BaseStepping::RealTime;
        last_measure_.clear();
        title_dirty_ = true;
    }

    // flush_collapse_error / decay state: BaseDirector's.

    ses_shell::AtomModel atom_;
    bool proj_ready_ = false;
    bool pending_measure_ = false;
};

}  // namespace ses_shell
