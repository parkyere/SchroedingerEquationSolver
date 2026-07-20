// std::complex used directly -- outside the reinvention boundary (only
// third-party libs are reinvented). Built -fcx-limited-range: naive formulas,
// no Annex G NaN fixups.
// EXPECT_EQ where values are exactly representable; EXPECT_DOUBLE_EQ for sqrt/div.

#include <complex>

#include <gtest/gtest.h>

// clang + libstdc++ cannot constant-fold std::complex __complex__ compound
// assignments (libc++/MSVC/GCC are fine) -> arithmetic pins run at runtime there.
#if defined(__clang__) && defined(__GLIBCXX__)
#define SES_COMPLEX_ARITH_CONSTEXPR const
#else
#define SES_COMPLEX_ARITH_CONSTEXPR constexpr
#endif

namespace {

using Cd = std::complex<double>;

TEST(Complex, DefaultConstructsToZero) {
    constexpr Cd z{};
    EXPECT_EQ(z.real(), 0.0);
    EXPECT_EQ(z.imag(), 0.0);
}

TEST(Complex, AggregateConstruction) {
    constexpr Cd z{3.0, -4.0};
    EXPECT_EQ(z.real(), 3.0);
    EXPECT_EQ(z.imag(), -4.0);
}

TEST(Complex, Addition) {
    SES_COMPLEX_ARITH_CONSTEXPR Cd s = Cd{1.0, 2.0} + Cd{3.0, -5.0};
    EXPECT_EQ(s.real(), 4.0);
    EXPECT_EQ(s.imag(), -3.0);
}

TEST(Complex, Subtraction) {
    SES_COMPLEX_ARITH_CONSTEXPR Cd d = Cd{1.0, 2.0} - Cd{3.0, -5.0};
    EXPECT_EQ(d.real(), -2.0);
    EXPECT_EQ(d.imag(), 7.0);
}

TEST(Complex, MultiplicationSatisfiesISquaredEqualsMinusOne) {
    constexpr Cd i{0.0, 1.0};
    SES_COMPLEX_ARITH_CONSTEXPR Cd ii = i * i;
    EXPECT_EQ(ii.real(), -1.0);
    EXPECT_EQ(ii.imag(), 0.0);
}

TEST(Complex, MultiplicationGeneralCase) {
    SES_COMPLEX_ARITH_CONSTEXPR Cd p = Cd{1.0, 2.0} * Cd{3.0, 4.0};
    EXPECT_EQ(p.real(), -5.0);
    EXPECT_EQ(p.imag(), 10.0);
}

TEST(Complex, ScalarMultiplicationFromBothSides) {
    SES_COMPLEX_ARITH_CONSTEXPR Cd l = 2.0 * Cd{1.0, -3.0};
    SES_COMPLEX_ARITH_CONSTEXPR Cd r = Cd{1.0, -3.0} * 2.0;
    EXPECT_EQ(l.real(), 2.0);
    EXPECT_EQ(l.imag(), -6.0);
    EXPECT_EQ(r.real(), 2.0);
    EXPECT_EQ(r.imag(), -6.0);
}

TEST(Complex, Conjugate) {
    constexpr Cd c = conj(Cd{3.0, -4.0});
    EXPECT_EQ(c.real(), 3.0);
    EXPECT_EQ(c.imag(), 4.0);
}

TEST(Complex, NormSquaredIsSquaredMagnitude) {
    constexpr double n = std::norm(Cd{3.0, -4.0});
    EXPECT_EQ(n, 25.0);
}

TEST(Complex, AbsIsMagnitude) {
    EXPECT_DOUBLE_EQ(abs(Cd{3.0, -4.0}), 5.0);
}

TEST(Complex, DivisionByComplex) {
    const Cd q = Cd{-5.0, 10.0} / Cd{3.0, 4.0};
    EXPECT_DOUBLE_EQ(q.real(), 1.0);
    EXPECT_DOUBLE_EQ(q.imag(), 2.0);
}

TEST(Complex, MultiplicationConjugateGivesRealNormSq) {
    constexpr Cd z{3.0, -4.0};
    SES_COMPLEX_ARITH_CONSTEXPR Cd zz = z * conj(z);
    EXPECT_EQ(zz.real(), 25.0);
    EXPECT_EQ(zz.imag(), 0.0);
}

}  // namespace
