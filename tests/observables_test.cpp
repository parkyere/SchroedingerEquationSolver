// RED: probability_in_range(psi, a, b) -- the absolute probability content
// of the half-open interval [a, b):
//
//     P = sum_{a <= x_i < b} |psi_i|^2 h
//
// Deliberately NOT scale-invariant (unlike the other observables): the
// tunneling readout T = P(right of barrier) is measured against the initial
// unit norm, so flux removed by the absorbing mask must reduce -- never
// inflate -- the report. Total over the whole box therefore equals norm_sq.

#include <gtest/gtest.h>

#include <complex>

import ses.field;
import ses.grid;
import ses.observables;
import ses.wavepacket;

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

}  // namespace
