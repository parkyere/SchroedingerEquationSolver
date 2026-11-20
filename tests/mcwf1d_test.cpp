// RED: 1D photon-loss MCWF step. Jump: even cat (+) -> odd cat (-); no-jump
// damping bleeds <n> at kappa (amplitude alpha e^{-kappa t/2}; hence /2 in tau).

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <vector>

import ses.mcwf1d;
import ses.observables;
import ses.potential;
import ses.propagator;
import ses.wavepacket;

namespace {

ses::Field1D cat(const ses::Grid1D& g, double omega, double x0, int sign) {
    const double sig = 1.0 / std::sqrt(2.0 * omega);
    ses::Field1D a = ses::gaussian_wavepacket(g, x0, sig, 0.0);
    ses::Field1D b = ses::gaussian_wavepacket(g, -x0, sig, 0.0);
    for (int i = 0; i < g.n; ++i) {
        a[i] += static_cast<double>(sign) * b[i];
    }
    ses::normalize(a);
    return a;
}

double overlap2(const ses::Field1D& a, const ses::Field1D& b) {
    return std::norm(ses::inner_product(a, b));
}

TEST(Mcwf1d, JumpFlipsTheCatParity) {
    const ses::Grid1D g{-12.0, 12.0, 512};
    const double omega = 1.0;
    const std::vector<double> v = ses::harmonic_potential(g, omega, 0.0);
    const ses::PhotonLossDamper damp{g, v, omega, 0.05, 0.01};
    ses::Field1D psi = cat(g, omega, 3.0, +1);
    // u = 0 forces the jump (<n> ~ alpha^2 = 4.5).
    const bool jumped = ses::photon_loss_step(psi, 0.0, damp);
    EXPECT_TRUE(jumped);
    EXPECT_GT(overlap2(psi, cat(g, omega, 3.0, -1)), 0.9);
    EXPECT_LT(overlap2(psi, cat(g, omega, 3.0, +1)), 0.05);
}

// Every check below runs at the scene's trap frequency (0.25) and at 2 as
// well as at 1: at omega = 1 a misplaced factor of omega (in dtau =
// kappa dt / (2 omega), in <n> = E / omega - 1/2, in the jump rate) is
// invisible. The grid scales with 1/sqrt(omega), so the three runs are the
// same problem in oscillator units; the coherent state has |alpha|^2 = 4.5.
class Mcwf1dOmega : public ::testing::TestWithParam<double> {
  protected:
    double omega() const { return GetParam(); }
    ses::Grid1D grid() const {
        const double half = 12.0 / std::sqrt(omega());
        return ses::Grid1D{-half, half, 512};
    }
    ses::Field1D coherent(const ses::Grid1D& g) const {
        const double sig = 1.0 / std::sqrt(2.0 * omega());
        return ses::gaussian_wavepacket(g, 3.0 / std::sqrt(omega()), sig, 0.0);
    }
};

// The jump fires iff u < kappa <n> dt, with <n> = |alpha|^2 = 4.5 here.
TEST_P(Mcwf1dOmega, JumpThresholdIsKappaTimesNTimesDt) {
    const ses::Grid1D g = grid();
    const double kappa = 0.5;
    const double dt = 0.01;
    const std::vector<double> v = ses::harmonic_potential(g, omega(), 0.0);
    const ses::PhotonLossDamper damp{g, v, omega(), kappa, dt};
    const double p_jump = kappa * 4.5 * dt;
    ses::Field1D below = coherent(g);
    EXPECT_TRUE(ses::photon_loss_step(below, 0.99 * p_jump, damp));
    ses::Field1D above = coherent(g);
    EXPECT_FALSE(ses::photon_loss_step(above, 1.01 * p_jump, damp));
}

// No-jump evolution of a coherent state: alpha -> alpha e^{-kappa t / 2}, so
// <n> falls as e^{-kappa t} exactly; only the two Trotter splittings are
// left (measured relative deviation of the ratio: 2e-5 or less at all three
// frequencies; the gate is 10x that).
TEST_P(Mcwf1dOmega, NoJumpDampingBleedsNAtKappa) {
    const ses::Grid1D g = grid();
    const double kappa = 0.5;
    const double dt = 0.01 / omega();  // the same step in oscillator units
    const std::vector<double> v = ses::harmonic_potential(g, omega(), 0.0);
    const ses::SplitOperator1D prop{g, v, dt};
    // dtau = kappa dt/(2 omega) is derived INSIDE the damper (the contract).
    const ses::PhotonLossDamper damp{g, v, omega(), kappa, dt};
    ses::Field1D psi = coherent(g);
    auto n_of = [&](const ses::Field1D& f) {
        return ses::mean_energy(f, v) / omega() - 0.5;
    };
    const double n0 = n_of(psi);
    ASSERT_NEAR(n0, 4.5, 1e-6);  // alpha^2 = x0^2 omega / 2
    const int steps = 400;
    for (int s = 0; s < steps; ++s) {  // u = 1: never jump
        prop.step(psi, 1);
        ses::photon_loss_step(psi, 1.0, damp);
    }
    const double ratio = n_of(psi) / n0;
    const double pred = std::exp(-kappa * steps * dt);
    std::printf("mcwf1d omega %.2f: <n> ratio %.6f vs e^{-kappa T} %.6f (rel %.2e)\n",
                omega(), ratio, pred, ratio / pred - 1.0);
    EXPECT_NEAR(ratio, pred, 2e-4 * pred);
}

INSTANTIATE_TEST_SUITE_P(TrapFrequency, Mcwf1dOmega,
                         ::testing::Values(0.25, 1.0, 2.0));

}  // namespace
