module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.doublewell1d_director;
export import ses.scenario.line1d_director;
import ses.spectrum1d;


// The double-well tunneling oscillation (the ammonia-inversion demo): the
// ground doublet psi_0 (symmetric) / psi_1 (antisymmetric) is split by dE,
// and psi_L = (psi_0 + psi_1)/sqrt(2) -- the "particle in the left well" --
// coherently transfers to the right well at t = pi/dE and back. dE depends
// EXPONENTIALLY on the barrier: the panel slider makes the oscillation
// crawl or race. The doublet comes from ses.spectrum1d (the radial FD
// solver); the license physics (splitting drives a full transfer through
// the real split-step propagator) is tests/solvable_wells_test.cpp.


export namespace ses_shell {

constexpr double kDw1dBox = 40.0;      // Bohr half-extent
constexpr int kDw1dPoints = 65536;
constexpr double kDw1dA = 6.0;         // well minima at +-a
constexpr double kDw1dBarrier = 0.12;  // boot barrier height (Ha)
constexpr double kDw1dBarrierMin = 0.04;
constexpr double kDw1dBarrierMax = 0.30;
constexpr double kDw1dDt = 0.04;
constexpr double kDw1dRScale = 25.0;
constexpr double kDw1dEScale = 65.0;   // barrier 0.12 -> ~8 Bohr tall
constexpr double kDw1dYClamp = 1e30;   // walls leave the frame honestly

class DoubleWell1DDirector final : public Line1DDirector, public DoubleWellApi {
public:
    DoubleWell1DDirector()
        : Line1DDirector(ses::Grid1D{-kDw1dBox, kDw1dBox, kDw1dPoints},
                         ses::double_well_potential(
                             ses::Grid1D{-kDw1dBox, kDw1dBox, kDw1dPoints},
                             kDw1dBarrier, kDw1dA),
                         kDw1dDt, kDw1dRScale, kDw1dEScale, kDw1dYClamp) {
        prepare_left();
    }

    DoubleWellApi* doublewell() override { return this; }

    // ---- DoubleWellApi ----
    double splitting() const override { return de_; }
    double p_left() const override { return p_left_; }
    double p_right() const override { return p_right_; }
    double barrier() const override { return vb_; }
    void set_barrier(double vb) override {
        vb_ = std::clamp(vb, kDw1dBarrierMin, kDw1dBarrierMax);
        set_potential(ses::double_well_potential(grid1d_, vb_, kDw1dA));
        prepare_left();  // a preparation demo: re-launch psi_L
    }

    double default_camera_azimuth() const override { return 0.3; }
    double default_camera_elevation() const override { return 0.25; }
    double default_camera_distance() const override { return 60.0; }

protected:
    const char* scene_name() const override {
        return "1D double well (tunneling oscillation)";
    }
    int steps_per_tick() const override { return 2; }

    std::string title_suffix() override {
        return strf("  Vb = %.2f Ha  dE = %.2e Ha  transfer T = pi/dE = %.0f "
                    "au  P_L = %.3f | P_R = %.3f",
                    vb_, de_, 3.14159265358979 / de_, p_left_, p_right_);
    }

    void after_batch() override {
        p_left_ = ses::probability_in_range(psi_, grid1d_.xmin, 0.0);
        p_right_ = ses::probability_in_range(psi_, 0.0, grid1d_.xmax);
    }

    void after_reset() override { after_batch(); }

private:
    // Solve the doublet and prepare the left-well state (psi_0 + psi_1)/
    // sqrt(2) -- both positive near xmin by the solver's sign convention,
    // so the sum concentrates LEFT.
    void prepare_left() {
        const std::vector<ses::Bound1D> s =
            ses::bound_states_1d(grid1d_, potential_, 2);
        de_ = s[1].energy - s[0].energy;
        ses::Field1D psi{grid1d_};
        for (int i = 0; i < grid1d_.n; ++i) {
            psi[i] = s[0].psi[i] + s[1].psi[i];
        }
        ses::normalize(psi);
        set_state(std::move(psi));
        after_batch();
        title_dirty_ = true;
    }

    double vb_ = kDw1dBarrier;
    double de_ = 0.0;
    double p_left_ = 0.0;
    double p_right_ = 0.0;
};

}  // namespace ses_shell
