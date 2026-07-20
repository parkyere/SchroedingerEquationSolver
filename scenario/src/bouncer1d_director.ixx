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


// Quantum bouncer: a particle on a hard floor under gravity, V = g z
// (z >= 0) with a steep linear wall below -- the GRANIT experiment's
// geometry (ultracold neutrons bounce on a mirror; the bound states are
// Airy functions, E_n = a_n (g^2/2)^{1/3} with -a_n the Airy zeros).
// [2] relaxes the Airy ground in-place (a 1D ITP is instant); [F] drops
// a packet from height and it bounces, dephases, and revives.
// CONTRACT: tests/bouncer1d_test.cpp + --selftest-bouncer.


export namespace ses_shell {

// Airy zeros a_1, a_2 (Ai(-a_n) = 0).
inline constexpr double kAiryZero1 = 2.33810741045977;
inline constexpr double kAiryZero2 = 4.08794944413097;

// E_n = a_n (g^2 / 2)^{1/3} for the ideal hard floor (m = hbar = 1).
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

// Sub-floor reach and dt sized for the WALL's real-time Trotter phase:
// the Airy tail only penetrates ~0.15 Bohr, but V dt at the box lip must
// stay well under a radian or the ground HEATS visibly (measured +0.8 Ha
// over ~4 s at dt = 0.01 with a -2 lip; the benzene dt rule).
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

class Bouncer1DDirector final : public Line1DDirector, public BouncerApi {
public:
    Bouncer1DDirector()
        : Line1DDirector(ses::Grid1D{kBo1dZLo, kBo1dZHi, kBo1dPoints},
                         bouncer_potential(
                             ses::Grid1D{kBo1dZLo, kBo1dZHi, kBo1dPoints},
                             kBo1dGrav, kBo1dWall),
                         kBo1dDt, kBo1dRScale, kBo1dEScale, 1e30) {
        relax_ground();
    }

    BouncerApi* bouncer() override { return this; }

    // ---- BouncerApi ----
    // Instant synchronous ITP anneal (a 2048-pt line costs ~0.2 s):
    // coarse settle, fine polish (the corral rule -- one big dtau alone
    // carries a visible Trotter bias).
    void relax_ground() override {
        const ses::ImaginaryTimePropagator1D coarse{grid1d_, potential_,
                                                    0.005};
        const ses::ImaginaryTimePropagator1D fine{grid1d_, potential_,
                                                  0.0005};
        ses::Field1D psi = ses::gaussian_wavepacket(grid1d_, 3.0, 1.5, 0.0);
        coarse.relax(psi, 3000);
        fine.relax(psi, 3000);
        set_state(std::move(psi));
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
        if (key == '2') {
            relax_ground();
            return true;
        }
        if (key == 'F') {
            drop();
            return true;
        }
        return false;
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
            "  g = %.1f  <H> = %.3f Ha  Airy E1 = %.3f (soft floor "
            "-g delta ~ -0.30)  T_bounce = %.1f au  keys: 2 ground / "
            "F drop",
            kBo1dGrav, energy(), airy_e1(),
            2.0 * std::sqrt(2.0 * kBo1dDropZ / kBo1dGrav));
    }

    void after_reset() override { relax_ground(); }
};

}  // namespace ses_shell
