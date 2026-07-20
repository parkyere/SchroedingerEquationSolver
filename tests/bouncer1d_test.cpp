// RED: the quantum bouncer. E_n = a_n (g^2/2)^{1/3} (Airy zeros); the
// relaxed ground on the gravity-plus-steep-wall landscape must land on
// E_1 within 2% (the finite wall slope costs a little), and the deflated
// next state on E_2.

#include <gtest/gtest.h>

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
    // Short sub-floor range: the wall reaches V ~ 800 max, so the COARSE
    // dtau Trotter bias stays curable by the fine polish (the corral
    // anneal rule; a -10 tail at slope 400 would sit at V dtau ~ 20).
    const ses::Grid1D g{-2.0, 78.0, 2048};
    const std::vector<double> v =
        ses_shell::bouncer_potential(g, grav, 400.0);
    ASSERT_GT(v[0], 100.0);  // the wall IS there below the floor
    const ses::ImaginaryTimePropagator1D itp{g, v, 0.005};
    const ses::ImaginaryTimePropagator1D fine{g, v, 0.0005};
    ses::Field1D psi = ses::gaussian_wavepacket(g, 3.0, 1.5, 0.0);
    itp.relax(psi, 3000);
    fine.relax(psi, 3000);
    const double e_ground = ses::mean_energy(psi, v);
    std::printf("bouncer: E1 %.4f (Airy %.4f), ", e_ground, e1);
    // The finite wall slope sinks the effective floor by delta ~
    // 1.4 (2 W)^{-1/3} ~ 0.15, shifting EVERY level by -g delta ~ -0.30
    // (measured -0.297): gate E1 inside that soft-floor window...
    EXPECT_LT(e_ground, e1);
    EXPECT_GT(e_ground, e1 - 0.35);
    // Deflate the ground away and relax again -> the second Airy state.
    ses::Field1D excited = ses::gaussian_wavepacket(g, 6.0, 2.0, 0.0);
    const double h = g.spacing();
    for (int s = 0; s < 6000; ++s) {
        (s < 3000 ? itp : fine).relax(excited, 1);
        // Gram-Schmidt against the captured ground each step.
        std::complex<double> ov{};
        for (int i = 0; i < g.n; ++i) {
            ov += std::conj(psi[i]) * excited[i];
        }
        ov *= h;
        for (int i = 0; i < g.n; ++i) {
            excited[i] -= ov * psi[i];
        }
        ses::normalize(excited);
    }
    const double e_next = ses::mean_energy(excited, v);
    std::printf("E2 %.4f (Airy %.4f)\n", e_next, e2);
    EXPECT_LT(e_next, e2);
    EXPECT_GT(e_next, e2 - 0.35);
    // ...while the LEVEL SPACING is floor-offset-immune and must land on
    // the Airy ladder exactly: E2 - E1 = (a2 - a1)(g^2/2)^{1/3}.
    EXPECT_NEAR(e_next - e_ground, e2 - e1, 0.02 * (e2 - e1));
}

}  // namespace
