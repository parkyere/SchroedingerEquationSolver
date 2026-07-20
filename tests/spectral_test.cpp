// FFT-bin -> physical wavenumber, NumPy fftfreq layout: k_j = 2pi j/L (j<n/2) else 2pi(j-n)/L.

#include <gtest/gtest.h>

#include <numbers>
#include <vector>

import ses.grid;
import ses.spectral;

namespace {

using ses::Grid1D;

constexpr double kTwoPi = 2.0 * std::numbers::pi;

TEST(Wavenumbers, MatchesFftfreqLayout) {
    const Grid1D g{0.0, 8.0, 8};
    const std::vector<double> k = ses::wavenumbers(g);
    ASSERT_EQ(k.size(), 8u);
    const double dk = kTwoPi / 8.0;
    EXPECT_DOUBLE_EQ(k[0], 0.0);
    EXPECT_DOUBLE_EQ(k[1], dk);
    EXPECT_DOUBLE_EQ(k[2], 2.0 * dk);
    EXPECT_DOUBLE_EQ(k[3], 3.0 * dk);
    EXPECT_DOUBLE_EQ(k[4], -4.0 * dk);  // Nyquist bin: negative branch
    EXPECT_DOUBLE_EQ(k[5], -3.0 * dk);
    EXPECT_DOUBLE_EQ(k[6], -2.0 * dk);
    EXPECT_DOUBLE_EQ(k[7], -dk);
}

// n=1 squashed axis (2D scenes' z) must be DC, not the j-n=-1 branch:
// spurious k=-2pi/L adds constant +k^2/2 to 3D energy (pi^2/2 at L=2 = corral phantom +4.9348).
TEST(Wavenumbers, SingleBinAxisIsDc) {
    const Grid1D g{-1.0, 1.0, 1};
    const std::vector<double> k = ses::wavenumbers(g);
    ASSERT_EQ(k.size(), 1u);
    EXPECT_DOUBLE_EQ(k[0], 0.0);
}

TEST(Wavenumbers, SpacingIsIndependentOfOffset) {
    const Grid1D g{-13.0, -5.0, 8};  // same L = 8
    const std::vector<double> k = ses::wavenumbers(g);
    const double dk = kTwoPi / 8.0;
    EXPECT_DOUBLE_EQ(k[1], dk);
    EXPECT_DOUBLE_EQ(k[7], -dk);
}

}  // namespace
