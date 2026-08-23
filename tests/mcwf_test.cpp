// RED: MCWF no-jump damping. On survival, scale excited amplitudes by
// exp(-gamma_n dt/2) then renormalize -- caller-side transform, mirrors GPU apply_mcwf_damping.

#include <complex>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
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

// Chained first-arrival jumps: cascade hops fire within ONE accumulated
// interval (single-jump truncation defers them; the error grows with
// time_scale, so a fast GPU at x16 distorts inter-orbital timing).

TEST(ChainDecayJumps, ZeroRatesFireNothingAndDrawNothing) {
    const std::vector<ses::RateChannel> ch{{1, 0, 0.7}};
    auto never = [] {
        ADD_FAILURE() << "no uniforms may be drawn";
        return 0.5;
    };
    EXPECT_TRUE(ses::chain_decay_jumps(ch, {1.0, 0.0}, 5.0, never).empty());
}

TEST(ChainDecayJumps, TwoLevelFirstArrivalMatchesTheExponential) {
    // R = gamma*pop = 0.5; t1 = -ln(u1)/R.
    const std::vector<ses::RateChannel> ch{{1, 0, 0.5}};
    {
        // u1 = e^{-1} -> t1 = 2 < dt=4 -> fires; destination has no outgoing
        // channel, so exactly one jump.
        std::vector<double> us{std::exp(-1.0), 0.0};
        std::size_t k = 0;
        auto u01 = [&] { return us[k++]; };
        const std::vector<int> fired =
            ses::chain_decay_jumps(ch, {0.0, 1.0}, 4.0, u01);
        ASSERT_EQ(fired.size(), 1u);
        EXPECT_EQ(fired[0], 0);
    }
    {
        // u1 = e^{-3} -> t1 = 6 > dt=4 -> survives.
        std::vector<double> us{std::exp(-3.0)};
        std::size_t k = 0;
        auto u01 = [&] { return us[k++]; };
        EXPECT_TRUE(ses::chain_decay_jumps(ch, {0.0, 1.0}, 4.0, u01).empty());
    }
}

TEST(ChainDecayJumps, CascadeHopsFireWithinOneInterval) {
    // 2->1 then 1->0, both gamma=1, dt=10; scripted arrivals t=1 then t=2.
    const std::vector<ses::RateChannel> ch{{2, 1, 1.0}, {1, 0, 1.0}};
    std::vector<double> us{std::exp(-1.0), 0.0, std::exp(-2.0), 0.0};
    std::size_t k = 0;
    auto u01 = [&] { return us[k++]; };
    const std::vector<int> fired =
        ses::chain_decay_jumps(ch, {0.0, 0.0, 1.0}, 10.0, u01);
    ASSERT_EQ(fired.size(), 2u);
    EXPECT_EQ(fired[0], 0);
    EXPECT_EQ(fired[1], 1);
}

TEST(ChainDecayJumps, CascadeStatisticsMatchTheTwoStageOracle) {
    // Equal rates: second arrival is Erlang(2): P(>=1) = 1-e^-x,
    // P(>=2) = 1-e^-x(1+x); x = 3.
    const std::vector<ses::RateChannel> ch{{2, 1, 1.0}, {1, 0, 1.0}};
    std::mt19937 rng(20260823u);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    auto u01 = [&] { return uni(rng); };
    const int n = 20000;
    int one = 0;
    int two = 0;
    for (int t = 0; t < n; ++t) {
        const std::size_t fired =
            ses::chain_decay_jumps(ch, {0.0, 0.0, 1.0}, 3.0, u01).size();
        one += fired >= 1 ? 1 : 0;
        two += fired >= 2 ? 1 : 0;
    }
    const double x = 3.0;
    EXPECT_NEAR(one / static_cast<double>(n), 1.0 - std::exp(-x), 0.02);
    EXPECT_NEAR(two / static_cast<double>(n),
                1.0 - std::exp(-x) * (1.0 + x), 0.02);
}

}  // namespace
