// RED: MCWF no-jump damping. On survival, scale excited amplitudes by
// exp(-gamma_n dt/2) then renormalize -- caller-side transform, mirrors GPU apply_mcwf_damping.

#include <complex>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>
import ses.decay;

namespace {


double pop(const std::vector<std::complex<double>>& c, int i) {
    return std::norm(c[static_cast<std::size_t>(i)]);
}

TEST(NoJumpDamping, IsIdentityOnAPureEigenstate) {
    const std::vector<std::complex<double>> c{{0.0, 0.0}, {0.6, -0.8}, {0.0, 0.0}};
    const std::vector<double> gamma{0.0, 0.13, 0.0};
    const std::vector<std::complex<double>> out =
        ses::nojump_damped_amplitudes(c, gamma, 2.0);
    EXPECT_NEAR(out[1].real(), 0.6, 1e-12);
    EXPECT_NEAR(out[1].imag(), -0.8, 1e-12);
    EXPECT_NEAR(pop(out, 0), 0.0, 1e-12);
    EXPECT_NEAR(pop(out, 2), 0.0, 1e-12);
}

TEST(NoJumpDamping, DrainsTheFasterDecayingComponentAndConservesNorm) {
    const double s = 1.0 / std::sqrt(2.0);
    const std::vector<std::complex<double>> c{{s, 0.0}, {s, 0.0}};
    const std::vector<double> gamma{0.0, 0.4};
    const std::vector<std::complex<double>> out =
        ses::nojump_damped_amplitudes(c, gamma, 1.0);
    EXPECT_GT(pop(out, 0), 0.5);
    EXPECT_LT(pop(out, 1), 0.5);
    EXPECT_NEAR(pop(out, 0) + pop(out, 1), 1.0, 1e-12);
    // Analytic no-jump: p0 = 1/(1+e^{-gamma dt}).
    EXPECT_NEAR(pop(out, 0), 1.0 / (1.0 + std::exp(-0.4)), 1e-12);
}

TEST(NoJumpDamping, GroundGrowsMonotonicallyTowardOne) {
    std::vector<std::complex<double>> c{{0.5, 0.0}, {std::sqrt(0.75), 0.0}};
    const std::vector<double> gamma{0.0, 0.25};
    double prev = pop(c, 0);
    for (int step = 0; step < 40; ++step) {
        c = ses::nojump_damped_amplitudes(c, gamma, 1.0);
        EXPECT_GT(pop(c, 0), prev - 1e-15);
        prev = pop(c, 0);
    }
    EXPECT_GT(pop(c, 0), 0.99);
}

TEST(NoJumpDamping, DegenerateShellKeepsRelativeWeights) {
    const std::vector<std::complex<double>> c{{0.6, 0.0}, {0.0, 0.8}};
    const std::vector<double> gamma{0.3, 0.3};
    const std::vector<std::complex<double>> out =
        ses::nojump_damped_amplitudes(c, gamma, 1.5);
    EXPECT_NEAR(pop(out, 0) / pop(out, 1), 0.36 / 0.64, 1e-12);
    EXPECT_NEAR(pop(out, 0) + pop(out, 1), 1.0, 1e-12);
}

// bound_survival_ratio contract: absorbed flux IS ionization; MCWF H_eff damping
// is NOT and must be backed out of the bound-survival product.

TEST(IonizationTally, NoLossLeavesSurvivalUnchanged) {
    EXPECT_NEAR(ses::bound_survival_ratio(1.0, 0.0, 1.0), 1.0, 1e-15);
}

TEST(IonizationTally, AbsorbedFluxCountsAsIonization) {
    // 10% of the norm left through the absorber this interval.
    EXPECT_NEAR(ses::bound_survival_ratio(0.9, 0.0, 1.0), 0.9, 1e-15);
}

TEST(IonizationTally, HeffDampingIsBackedOutNotCountedAsIonization) {
    // L=0.05 damping, no absorption -> survival stays 1 (damping is not escape).
    const double L = 0.05;
    EXPECT_NEAR(ses::bound_survival_ratio(1.0 - L, L, 1.0), 1.0, 1e-15);
}

TEST(IonizationTally, IsolatesAbsorptionWhenBothActed) {
    // Absorber leaves 0.95, then damping removes L=0.05 of the original norm ->
    // post=0.90; backing out L recovers the absorbed survival 0.95.
    const double L = 0.05;
    EXPECT_NEAR(ses::bound_survival_ratio(0.95 - L, L, 1.0), 0.95, 1e-15);
}

}  // namespace
