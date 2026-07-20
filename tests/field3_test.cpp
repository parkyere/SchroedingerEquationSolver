// RED spec: ||psi||^2 = sum_ijk |psi_ijk|^2 * hx hy hz

#include <complex>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
import ses.grid;
import ses.field;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;

const Grid3D kSmall{Grid1D{0.0, 8.0, 16}, Grid1D{0.0, 4.0, 8}, Grid1D{-1.0, 1.0, 4}};

TEST(Field3D, SizeMatchesGridAndZeroInit) {
    const Field3D f{kSmall};
    EXPECT_EQ(f.size(), 16 * 8 * 4);
    EXPECT_EQ(f(5, 3, 2).real(), 0.0);
    EXPECT_EQ(f(5, 3, 2).imag(), 0.0);
}

TEST(Field3D, TripleIndexMatchesFlatLayout) {
    Field3D f{kSmall};
    f(3, 2, 1) = std::complex<double>{1.5, -2.5};
    const int flat = kSmall.flat(3, 2, 1);
    EXPECT_EQ(f.data()[static_cast<std::size_t>(flat)].real(), 1.5);
    EXPECT_EQ(f.data()[static_cast<std::size_t>(flat)].imag(), -2.5);
}

TEST(Field3D, NormIncludesCellVolume) {
    // psi=1: (16*8*4 cells) * (cell-vol 0.125) = 64 exact
    Field3D f{kSmall};
    for (int i = 0; i < f.size(); ++i) {
        f.data()[static_cast<std::size_t>(i)] = std::complex<double>{1.0, 0.0};
    }
    EXPECT_EQ(norm_sq(f), 64.0);
}

TEST(Field3D, SeparableGaussianHasUnitNorm) {
    // Product of three continuum-normalized 1D Gaussians (sigma = 1).
    const Grid1D axis{-8.0, 8.0, 32};
    const Grid3D g{axis, axis, axis};
    Field3D f{g};
    const double amp = std::pow(2.0 * std::numbers::pi, -0.25);
    auto env = [&](double u) { return amp * std::exp(-u * u / 4.0); };
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double val = env(g.x.coord(i)) * env(g.y.coord(j)) * env(g.z.coord(k));
                f(i, j, k) = std::complex<double>{val, 0.0};
            }
        }
    }
    EXPECT_NEAR(norm_sq(f), 1.0, 1e-10);
}

TEST(Field3D, NormalizeMakesUnitNorm) {
    Field3D f{kSmall};
    for (int i = 0; i < f.size(); ++i) {
        f.data()[static_cast<std::size_t>(i)] =
            std::complex<double>{0.01 * i, -0.02 * i};
    }
    normalize(f);
    EXPECT_NEAR(norm_sq(f), 1.0, 1e-12);
}

}  // namespace
