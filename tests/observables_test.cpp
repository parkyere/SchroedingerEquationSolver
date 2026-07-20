// probability_in_range(psi, a, b) = sum_{a <= x_i < b} |psi_i|^2 h, half-open
// [a, b). Deliberately NOT scale-invariant: tunneling T is read against the
// initial unit norm, so absorbed flux must shrink -- never inflate -- the total.
// energy_variance(psi, V) = <H^2> - <H>^2, scale-invariant.

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
    // h = 1, coords 0..7.
    const ses::Grid1D g{0.0, 8.0, 8};
    ses::Field1D f{g};
    f[2] = std::complex<double>{1.0, 0.0};
    f[5] = std::complex<double>{0.0, 2.0};
    EXPECT_DOUBLE_EQ(ses::probability_in_range(f, 2.0, 3.0), 1.0);
    EXPECT_DOUBLE_EQ(ses::probability_in_range(f, 5.0, 6.0), 4.0);
    EXPECT_DOUBLE_EQ(ses::probability_in_range(f, 3.0, 5.0), 0.0);
    EXPECT_DOUBLE_EQ(ses::probability_in_range(f, 0.0, 8.0), 5.0);
    EXPECT_DOUBLE_EQ(ses::probability_in_range(f, 2.0, 2.0), 0.0);
    EXPECT_DOUBLE_EQ(ses::probability_in_range(f, 0.0, 2.0), 0.0);
}

TEST(ProbabilityInRange, TotalOverTheBoxIsNormSq) {
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

    // equal two-level mix: Var = (E1 - E0)^2 / 4
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
    // Centered on grid point x=0: mirror coords are bitwise negatives, so halves
    // agree to round-off. Unpaired wall point x=-16 has envelope ~exp(-32), negligible.
    const ses::Grid1D g{-16.0, 16.0, 128};  // h = 0.25
    const ses::Field1D f = ses::gaussian_wavepacket(g, 0.0, 2.0, 1.3);
    const double left = ses::probability_in_range(f, -16.0, 0.0);
    const double right = ses::probability_in_range(f, 0.25, 16.0);
    EXPECT_NEAR(left, right, 1e-14);
}

// LOCK: exact high-E eigenstate must stay at round-off. Naive <H^2> - <H>^2 has
// an E^2-scaled cancellation floor (~5e-6, passes 1e-8 only by MSVC luck); the
// residual form ||(H - <H>) psi||^2 has no floor -- pinned for every compiler.
TEST(EnergyVariance, HighEnergyEigenstateStaysAtRoundOff) {
    const ses::Grid1D g{-100.0, 100.0, 65536};  // the harmonic1d scene grid
    const double w = 4.0;
    const std::vector<double> v = ses::harmonic_potential(g, w);
    const ses::Field1D psi = ses::ho_eigenstate(g, w, 2500);  // E = 10002 Ha
    EXPECT_LT(ses::energy_variance(psi, v), 1e-8);
}

}  // namespace
