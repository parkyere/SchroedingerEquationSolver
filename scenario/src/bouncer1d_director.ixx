module;
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.bouncer1d_director;
export import ses.scenario.line1d_director;
import ses.imaginary_time;
import ses.observables;
import ses.wavepacket;


// Quantum bouncer (GRANIT): gravity + mirror wall, Airy bound states.
// CONTRACT: tests/bouncer1d_test.cpp + --selftest-bouncer.


export namespace ses_shell {

// Ai(-a_n) = 0.
inline constexpr double kAiryZero1 = 2.33810741045977;
inline constexpr double kAiryZero2 = 4.08794944413097;

// Ideal hard-floor oracle (m = hbar = 1).
inline double bouncer_energy(double g, double a_n) {
    return a_n * std::cbrt(g * g / 2.0);
}

// V = g z above the floor, steep linear wall (slope `wall`) below --
// continuous at z = 0, no Gibbs step.
inline std::vector<double> bouncer_potential(const ses::Grid1D& g,
                                             double grav, double wall) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double z = g.coord(i);
        v[static_cast<std::size_t>(i)] = z >= 0.0 ? grav * z : -wall * z;
    }
    return v;
}

// dt sized by the wall's Trotter phase, not resolution: V*dt at the box
// lip must stay << 1 rad or the ground heats (benzene dt rule).
constexpr double kBo1dZLo = -1.0;
constexpr double kBo1dZHi = 79.0;
constexpr int kBo1dPoints = 2048;
constexpr double kBo1dDt = 0.002;
constexpr double kBo1dGrav = 2.0;
constexpr double kBo1dWall = 400.0;
constexpr double kBo1dDropZ = 40.0;
constexpr double kBo1dDropSigma = 2.0;
constexpr double kBo1dRScale = 150.0;
constexpr double kBo1dEScale = 0.15;
constexpr int kBo1dStepsPerTick = 20;
// two-stage ITP anneal (fine polish clears the coarse Trotter bias).
constexpr double kBo1dItpCoarseDtau = 0.005;
constexpr double kBo1dItpFineDtau = 0.0005;
constexpr int kBo1dItpIters = 3000;

class Bouncer1DDirector final : public Line1DDirector, public BouncerApi {
public:
    Bouncer1DDirector()
        : Line1DDirector(scene_grid(),
                         bouncer_potential(scene_grid(), kBo1dGrav, kBo1dWall),
                         kBo1dDt, kBo1dRScale, kBo1dEScale, kNoYClamp) {
        relax_ground();
    }

    BouncerApi* bouncer() override { return this; }

    // Two-stage ITP anneal: fine polish clears the coarse dtau's Trotter
    // bias (corral rule).
    void relax_ground() override {
        const ses::ImaginaryTimePropagator1D coarse{grid1d_, potential_,
                                                    kBo1dItpCoarseDtau};
        const ses::ImaginaryTimePropagator1D fine{grid1d_, potential_,
                                                  kBo1dItpFineDtau};
        ses::Field1D psi = ses::gaussian_wavepacket(grid1d_, 3.0, 1.5, 0.0);
        coarse.relax(psi, kBo1dItpIters);
        fine.relax(psi, kBo1dItpIters);
        set_state(std::move(psi));
        e1_soft_ = energy();  // measured soft-floor E1 (title truth)
        sim_time_ = 0.0;
        pending_steps_ = 0;
        title_dirty_ = true;
    }
    void drop() override {
        set_state(ses::gaussian_wavepacket(grid1d_, kBo1dDropZ,
                                           kBo1dDropSigma, 0.0));
        sim_time_ = 0.0;
        pending_steps_ = 0;
        title_dirty_ = true;
    }
    double energy() const override {
        return ses::mean_energy(psi_, potential_);
    }
    double airy_e1() const override {
        return bouncer_energy(kBo1dGrav, kAiryZero1);
    }

    bool handle_key(char key) override {
        switch (key) {
            case '2': relax_ground(); return true;
            case 'F': drop(); return true;
            default: return false;
        }
    }

    double default_camera_azimuth() const override { return 0.25; }
    double default_camera_elevation() const override { return 0.22; }
    double default_camera_distance() const override { return 60.0; }

protected:
    const char* scene_name() const override {
        return "Quantum bouncer (gravity + mirror, GRANIT)";
    }
    int steps_per_tick() const override { return kBo1dStepsPerTick; }

    std::string title_suffix() override {
        return strf(
            "  g = {:.1f}  <H> = {:.3f} eV  Airy E1 = {:.3f} eV (soft floor "
            "{:+.2f} eV)  T_bounce = {:.1f} au  keys: 2 ground / "
            "F drop",
            kBo1dGrav, energy() * kHaToEv, airy_e1() * kHaToEv,
            (e1_soft_ - airy_e1()) * kHaToEv,
            2.0 * std::sqrt(2.0 * kBo1dDropZ / kBo1dGrav));
    }

    void after_reset() override { relax_ground(); }

private:
    static ses::Grid1D scene_grid() {
        return ses::Grid1D{kBo1dZLo, kBo1dZHi, kBo1dPoints};
    }

    double e1_soft_ = 0.0;  // relaxed <H>: soft-floor E1
};

}  // namespace ses_shell
