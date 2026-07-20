// RED: probability_in_range(psi, a, b) -- the absolute probability content
// of the half-open interval [a, b):
//
//     P = sum_{a <= x_i < b} |psi_i|^2 h
//
// Deliberately NOT scale-invariant (unlike the other observables): the
// tunneling readout T = P(right of barrier) is measured against the initial
// unit norm, so flux removed by the absorbing mask must reduce -- never
// inflate -- the report. Total over the whole box therefore equals norm_sq.
//
// RED: energy_variance(psi, V) -- Var(H) = <H^2> - <H>^2, scale-invariant,
// H psi built spectrally (T in k-space) + V in real space. The honest
// eigenstate discriminator: zero (to round-off) iff psi is an H eigenstate,
// and exactly (E1 - E0)^2 / 4 on an equal two-level superposition -- no
// measurement, no basis bookkeeping.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>

import ses.field;
import ses.grid;
import ses.observables;
import ses.potential;
import ses.wavepacket;
import ses.ladder;

namespace {

TEST(ProbabilityInRange, DefinitionOnHandmadeField) {
    // h = 1, coords 0..7; only two cells occupied.
    const ses::Grid1D g{0.0, 8.0, 8};
    ses::Field1D f{g};
    f[2] = std::complex<double>{1.0, 0.0};
    f[5] = std::complex<double>{0.0, 2.0};
    EXPECT_DOUBLE_EQ(ses::probability_in_range(f, 2.0, 3.0), 1.0);
    EXPECT_DOUBLE_EQ(ses::probability_in_range(f, 5.0, 6.0), 4.0);
    EXPECT_DOUBLE_EQ(ses::probability_in_range(f, 3.0, 5.0), 0.0);
    EXPECT_DOUBLE_EQ(ses::probability_in_range(f, 0.0, 8.0), 5.0);
    // Half-open: the lower bound is included, the upper excluded.
    EXPECT_DOUBLE_EQ(ses::probability_in_range(f, 2.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(ses::probability_in_range(f, 0.0, 2.0), 0.0);
}

TEST(ProbabilityInRange, TotalOverTheBoxIsNormSq) {
    // Absolute, not scale-invariant: the whole-box sum tracks the field's
    // actual norm (2x amplitude -> 4x probability).
    const ses::Grid1D g{-16.0, 16.0, 128};
    ses::Field1D f = ses::gaussian_wavepacket(g, 1.0, 2.0, 0.7);
    EXPECT_NEAR(ses::probability_in_range(f, -16.0, 16.0), 1.0, 1e-12);
    for (int i = 0; i < f.size(); ++i) {
        f[i] *= 2.0;
    }
    EXPECT_NEAR(ses::probability_in_range(f, -16.0, 16.0), ses::norm_sq(f), 1e-12);
    EXPECT_NEAR(ses::probability_in_range(f, -16.0, 16.0), 4.0, 1e-12);
}

TEST(ProbabilityInRange, AdjacentHalfOpenIntervalsTileExactly) {
    const ses::Grid1D g{-16.0, 16.0, 128};
    const ses::Field1D f = ses::gaussian_wavepacket(g, -3.0, 1.5, 0.0);
    const double left = ses::probability_in_range(f, -16.0, -2.0);
    const double mid = ses::probability_in_range(f, -2.0, 5.0);
    const double right = ses::probability_in_range(f, 5.0, 16.0);
    EXPECT_DOUBLE_EQ(left + mid + right,
                     ses::probability_in_range(f, -16.0, 16.0));
}

TEST(EnergyVariance, VanishesOnEigenstatesAndIsExactOnTwoLevelMixes) {
    // HO at omega = 0.25 on the ladder-clean grid; psi_0 is the exact
    // Gaussian, psi_1 = sqrt(2 omega) x psi_0 analytically.
    const double omega = 0.25;
    const ses::Grid1D g{-20.0, 20.0, 256};
    const std::vector<double> v = ses::harmonic_potential(g, omega);
    const double sigma = 1.0 / std::sqrt(2.0 * omega);
    ses::Field1D psi0 = ses::gaussian_wavepacket(g, 0.0, sigma, 0.0);
    EXPECT_NEAR(ses::energy_variance(psi0, v), 0.0, 1e-10);

    ses::Field1D psi1{g};
    for (int i = 0; i < g.n; ++i) {
        psi1[i] = std::sqrt(2.0 * omega) * g.coord(i) * psi0[i];
    }
    ses::normalize(psi1);
    EXPECT_NEAR(ses::energy_variance(psi1, v), 0.0, 1e-10);
    EXPECT_NEAR(ses::mean_energy(psi1, v), 1.5 * omega, 1e-10);

    // (|0> + |1>)/sqrt(2): <H> = omega, Var = (E1 - E0)^2 / 4 = omega^2/4.
    ses::Field1D mix{g};
    for (int i = 0; i < g.n; ++i) {
        mix[i] = psi0[i] + psi1[i];
    }
    ses::normalize(mix);
    EXPECT_NEAR(ses::mean_energy(mix, v), omega, 1e-10);
    EXPECT_NEAR(ses::energy_variance(mix, v), omega * omega / 4.0, 1e-10);
}

TEST(EnergyVariance, IsScaleInvariant) {
    const ses::Grid1D g{-16.0, 16.0, 128};
    const std::vector<double> v = ses::harmonic_potential(g, 0.5);
    ses::Field1D f = ses::gaussian_wavepacket(g, 1.0, 1.3, 0.4);
    const double var1 = ses::energy_variance(f, v);
    for (int i = 0; i < f.size(); ++i) {
        f[i] *= 3.0;
    }
    EXPECT_NEAR(ses::energy_variance(f, v), var1, 1e-9 * (1.0 + var1));
    EXPECT_GT(var1, 0.0);  // a displaced squeezed packet is no eigenstate
}

TEST(ProbabilityInRange, SymmetricPacketSplitsEvenlyAboutItsCenter) {
    // Packet centered on the grid point x = 0: mirror coords are bitwise
    // negatives, so the two half sums agree to round-off. The left half
    // includes the unpaired wall point x = -16, where the envelope is
    // ~exp(-32) -- far below double resolution here.
    const ses::Grid1D g{-16.0, 16.0, 128};  // h = 0.25
    const ses::Field1D f = ses::gaussian_wavepacket(g, 0.0, 2.0, 1.3);
    const double left = ses::probability_in_range(f, -16.0, 0.0);
    const double right = ses::probability_in_range(f, 0.25, 16.0);
    EXPECT_NEAR(left, right, 1e-14);
}

// LOCK: Var(H) of an EXACT high-energy eigenstate stays at round-off on
// the SCENE's 64k grid at E = 10002 Ha. The naive <H^2> - <H>^2 form has an
// E^2-scaled cancellation floor (upper bound ~ 5e-6 here; it happened to
// stay under the 1e-8 gate on THIS MSVC build, but the margin is platform
// luck) -- the residual form ||(H - <H>) psi||^2 has no floor at all, and
// this test pins that for every future compiler.
TEST(EnergyVariance, HighEnergyEigenstateStaysAtRoundOff) {
    const ses::Grid1D g{-100.0, 100.0, 65536};  // the harmonic1d scene grid
    const double w = 4.0;
    const std::vector<double> v = ses::harmonic_potential(g, w);
    const ses::Field1D psi = ses::ho_eigenstate(g, w, 2500);  // E = 10002 Ha
    EXPECT_LT(ses::energy_variance(psi, v), 1e-8);
}

}  // namespace
