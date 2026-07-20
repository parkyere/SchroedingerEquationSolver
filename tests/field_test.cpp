// Contract: discrete norm ||psi||^2 = sum_i |psi_i|^2 * h; grid weight h mandatory.

#include <complex>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
import ses.grid;
import ses.field;

namespace {

using ses::Field1D;
using ses::Grid1D;

TEST(Field1D, SizeMatchesGrid) {
    const Field1D f{Grid1D{0.0, 8.0, 16}};
    EXPECT_EQ(f.size(), 16);
    EXPECT_EQ(f.grid().n, 16);
}

TEST(Field1D, InitializesToZero) {
    const Field1D f{Grid1D{0.0, 8.0, 16}};
    for (int i = 0; i < f.size(); ++i) {
        EXPECT_EQ(f[i].real(), 0.0);
        EXPECT_EQ(f[i].imag(), 0.0);
    }
}

TEST(Field1D, ElementReadWrite) {
    Field1D f{Grid1D{0.0, 8.0, 16}};
    f[3] = std::complex<double>{1.5, -2.5};
    EXPECT_EQ(f[3].real(), 1.5);
    EXPECT_EQ(f[3].imag(), -2.5);
}

TEST(Field1D, NormSqIncludesGridWeight) {
    // h=0.5 exact in binary; 16 * 0.5 = 8 exactly, so EXPECT_EQ is valid.
    Field1D f{Grid1D{0.0, 8.0, 16}};
    for (int i = 0; i < f.size(); ++i) {
        f[i] = std::complex<double>{1.0, 0.0};
    }
    EXPECT_EQ(norm_sq(f), 8.0);
}

TEST(Field1D, ContinuumNormalizedGaussianHasUnitDiscreteNorm) {
    // Continuum Gaussian integrates to 1; tails ~e^-128 keep the Riemann sum tight to 1e-12.
    const Grid1D g{-16.0, 16.0, 256};
    Field1D f{g};
    const double s = 1.0;
    const double amp = std::pow(2.0 * std::numbers::pi * s * s, -0.25);
    for (int i = 0; i < f.size(); ++i) {
        const double x = g.coord(i);
        f[i] = std::complex<double>{amp * std::exp(-x * x / (4.0 * s * s)), 0.0};
    }
    EXPECT_NEAR(norm_sq(f), 1.0, 1e-12);
}

TEST(Field1D, NormalizeMakesUnitNorm) {
    Field1D f{Grid1D{-4.0, 4.0, 32}};
    for (int i = 0; i < f.size(); ++i) {
        f[i] = std::complex<double>{0.3 * i, -0.7 * i};
    }
    normalize(f);
    EXPECT_NEAR(norm_sq(f), 1.0, 1e-12);
}

}  // namespace
