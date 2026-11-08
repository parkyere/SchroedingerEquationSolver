// SpinDirector contracts (needs the GPU-layer library ses_scenario).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numbers>

import ses.scenario.spin_director;

namespace {

constexpr double kPi = std::numbers::pi;
constexpr double kTickDt = ses_shell::kSpStepsPerTick * ses_shell::kSpDt;

// The RF drive must co-rotate with the Larmor precession whatever way B
// points (SpinApi: "B free to point anywhere", "co-rotating Rabi drive at
// omega_L"). The boot spin (+x) precesses a quarter turn, so it is then
// transverse to B AND to the drive's start axis; RF goes on and runs one
// Rabi quarter period pi / (2 Omega_R). Returns the largest |<s>.B-hat| seen:
// ~1 for a resonant drive, small for a counter-rotating one.
double rabi_quarter_peak(double bx, double by, double bz) {
    ses_shell::SpinDirector d;
    ses_shell::SpinApi* s = d.spin();
    s->set_b(0, bx);
    s->set_b(1, by);
    s->set_b(2, bz);
    const double w = std::sqrt(bx * bx + by * by + bz * bz);
    const auto run = [&d](double t) {
        const long ticks = std::lround(t / kTickDt);
        for (long k = 0; k < ticks; ++k) {
            d.tick();
            d.run_frame();
        }
    };
    run(0.5 * kPi / w);  // quarter Larmor turn
    s->toggle_rf();
    double peak = 0.0;
    const long ticks = std::lround(0.5 * kPi / ses_shell::kSpRfB1 / kTickDt);
    for (long k = 0; k < ticks; ++k) {
        d.tick();
        d.run_frame();
        const double along =
            (s->bloch_x() * bx + s->bloch_y() * by + s->bloch_z() * bz) / w;
        peak = std::max(peak, std::abs(along));
    }
    return peak;
}

TEST(SpinDirector, RfDriveIsResonantForAnyFieldDirection) {
    EXPECT_GT(rabi_quarter_peak(0.0, 0.0, 0.5), 0.95) << "B along +z";
    EXPECT_GT(rabi_quarter_peak(0.0, 0.0, -0.5), 0.95) << "B along -z";
    EXPECT_GT(rabi_quarter_peak(0.0, 0.5, 0.0), 0.95) << "B along +y";
    EXPECT_GT(rabi_quarter_peak(0.3, -0.25, -0.3), 0.95) << "oblique B";
}

// The Bloch arrow and <s> follow every state change, not only time steps:
// a Measure, a pulse or a Reset made while paused must show at once.
TEST(SpinDirector, BlochReadoutFollowsInstantOperations) {
    ses_shell::SpinDirector d;
    ses_shell::SpinApi* s = d.spin();
    d.measure_now();  // boot +x, B along +z: collapses onto +-z
    EXPECT_NEAR(std::abs(s->bloch_z()), 1.0, 1e-9);
    EXPECT_NEAR(s->bloch_x(), 0.0, 1e-9);
    s->pulse(ses_shell::Pulse::Half);  // +-z tipped into the x = 0 plane
    EXPECT_NEAR(std::abs(s->bloch_y()), 1.0, 1e-9);
    d.reset_simulation();  // back to the boot +x
    EXPECT_NEAR(s->bloch_x(), 1.0, 1e-9);
}

}  // namespace
