// RED: quantum bouncer. E_n = a_n (g^2/2)^{1/3}, Airy zeros a_n; relaxed
// levels land on the Airy ladder.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <vector>

import ses.scenario.bouncer1d_director;
import ses.field;
import ses.grid;
import ses.imaginary_time;
import ses.observables;
import ses.wavepacket;

namespace {

TEST(Bouncer1d, RelaxedStatesLandOnTheAiryLadder) {
    const double grav = 2.0;
    const double e1 =
        ses_shell::bouncer_energy(grav, ses_shell::kAiryZero1);
    const double e2 =
        ses_shell::bouncer_energy(grav, ses_shell::kAiryZero2);
    EXPECT_NEAR(e1, 2.33810741045977 * std::cbrt(grav * grav / 2.0), 1e-9);
    // Coarse-dtau Trotter bias stays curable by the fine polish since the
    // wall V is bounded (corral anneal rule).
    const ses::Grid1D g{-2.0, 78.0, 2048};
    const std::vector<double> v =
        ses_shell::bouncer_potential(g, grav, 400.0);
    ASSERT_GT(v[0], 100.0);
    const ses::ImaginaryTimePropagator1D itp{g, v, 0.005};
    const ses::ImaginaryTimePropagator1D fine{g, v, 0.0005};
    ses::Field1D psi = ses::gaussian_wavepacket(g, 3.0, 1.5, 0.0);
    itp.relax(psi, 3000);
    fine.relax(psi, 3000);
    const double e_ground = ses::mean_energy(psi, v);
    std::printf("bouncer: E1 %.4f (Airy %.4f), ", e_ground, e1);
    // Finite wall slope shifts every level by ~ -0.30 (soft-floor offset);
    // gate E1 within that window.
    EXPECT_LT(e_ground, e1);
    EXPECT_GT(e_ground, e1 - 0.35);
    ses::Field1D excited = ses::gaussian_wavepacket(g, 6.0, 2.0, 0.0);
    for (int s = 0; s < 6000; ++s) {
        (s < 3000 ? itp : fine).relax(excited, 1);
        // Gram-Schmidt against the captured ground each step.
        const std::complex<double> ov = ses::inner_product(psi, excited);
        for (int i = 0; i < g.n; ++i) {
            excited[i] -= ov * psi[i];
        }
        ses::normalize(excited);
    }
    const double e_next = ses::mean_energy(excited, v);
    std::printf("E2 %.4f (Airy %.4f)\n", e_next, e2);
    EXPECT_LT(e_next, e2);
    EXPECT_GT(e_next, e2 - 0.35);
    // Level spacing is floor-offset-immune: E2 - E1 = (a2 - a1)(g^2/2)^{1/3}.
    EXPECT_NEAR(e_next - e_ground, e2 - e1, 0.02 * (e2 - e1));
}

// The scene's ground is deterministic (fixed potential, fixed seed): Reset and
// key 2 must hand back exactly the state the scene booted with.
TEST(Bouncer1dDirector, ResetAndKeyTwoRestoreTheBootGround) {
    ses_shell::Bouncer1DDirector d;
    ses_shell::BouncerApi* bo = d.bouncer();
    ASSERT_NE(bo, nullptr);
    const double e0 = bo->energy();
    bo->drop();
    ASSERT_NE(bo->energy(), e0);
    d.reset_simulation();
    EXPECT_EQ(bo->energy(), e0);
    bo->drop();
    EXPECT_TRUE(d.handle_key('2'));
    EXPECT_EQ(bo->energy(), e0);
}

// Real-time evolution at the scene's own step. The annealed ground is the
// ground of H, not of the Strang propagator, and the 400 Ha mirror wall makes
// the two differ visibly: <H> wobbles but must not drift. Measured over
// 128 au: ground max |dE| 2.3e-2 Ha (0.85 %), dropped packet 3.7e-3 Ha
// (5e-5), both bounded. The gates are ~2x that -- a regression lock on the
// Trotter error (it grows as dt^2 and with the wall height), covering the
// packet's fall and its first bounce.
TEST(Bouncer1dDirector, EvolutionKeepsEnergyWithinTheTrotterWobble) {
    ses_shell::Bouncer1DDirector d;
    ses_shell::BouncerApi* bo = d.bouncer();
    ASSERT_NE(bo, nullptr);
    const auto worst_change = [&d, bo](int ticks) {
        const double e0 = bo->energy();
        double worst = 0.0;
        for (int k = 0; k < ticks; ++k) {
            d.tick();
            d.run_frame();
            worst = std::max(worst, std::abs(bo->energy() - e0));
        }
        return worst;
    };
    EXPECT_LT(worst_change(200), 5e-2) << "ground state, 8 au";
    bo->drop();
    EXPECT_LT(worst_change(200), 1e-2) << "dropped packet, 8 au (fall + bounce)";
}

}  // namespace
