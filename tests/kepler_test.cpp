// RED: the Kepler packet seed. Circular-state pair table straight off the
// manifold spec; coefficients normalized, PURE m = +l (no m = -l
// admixture through the cos/sin pairing); and under the exact -1/(2n^2)
// spectrum the angular coherence <e^{i phi}> proxy is localized and its
// phase advances CCW at the Kepler rate ~ 1/n_bar^3 (the low-n tail skews
// the weighted mean fast of n_bar^-3, hence the generous window).

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>

import ses.scenario.kepler_seed;
import ses.measurement;

namespace {

TEST(KeplerSeed, CircularPairTableMatchesTheManifold) {
    const auto pairs = ses_shell::kepler_pairs();
    ASSERT_EQ(pairs.size(), 5u);
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        const ses_shell::KeplerPair& p = pairs[i];
        EXPECT_EQ(p.n, static_cast<int>(i) + 2);
        const StateSpec& sc = kStateSpec[p.idx_cos];
        const StateSpec& ss = kStateSpec[p.idx_sin];
        EXPECT_EQ(sc.l, p.n - 1);          // circular: l = n - 1
        EXPECT_EQ(sc.m, sc.l);             // cos entry: m = +l
        EXPECT_EQ(ss.level, sc.level);     // sin partner: same radial level
        EXPECT_EQ(ss.m, -sc.l);            // ...with m = -l
        EXPECT_EQ(state_n(p.idx_cos), p.n);
    }
}

TEST(KeplerSeed, CoefficientsAreNormalizedAndPureMPlus) {
    const auto c = ses_shell::kepler_coefficients(4.5, 1.0);
    ASSERT_EQ(static_cast<int>(c.size()), kNumStates);
    double sum = 0.0;
    int nonzero = 0;
    for (const auto& z : c) {
        sum += std::norm(z);
        nonzero += std::norm(z) > 1e-18 ? 1 : 0;
    }
    EXPECT_NEAR(sum, 1.0, 1e-12);
    EXPECT_EQ(nonzero, 10);  // 5 circular pairs, cos + sin entries only
    for (const ses_shell::KeplerPair& p : ses_shell::kepler_pairs()) {
        const ses::SignedM a =
            ses::signed_m_amplitudes(c[p.idx_cos], c[p.idx_sin]);
        EXPECT_LT(std::abs(a.minus), 1e-12);  // no m = -l admixture
        EXPECT_GT(std::abs(a.plus), 0.0);
    }
}

TEST(KeplerSeed, PacketOrbitsCcwAtTheKeplerRate) {
    const double n_bar = 4.5;
    const auto c = ses_shell::kepler_coefficients(n_bar, 1.0);
    const auto pairs = ses_shell::kepler_pairs();
    // Angular-coherence proxy at unit radial overlaps: adjacent circular
    // states differ by Delta m = 1, so <e^{i phi}> ~ sum_n a_{n+1}^* a_n
    // with each a_n(t) = a_n e^{-i E_n t}, E_n = -1/(2 n^2) exact.
    auto coherence = [&](double t) {
        std::complex<double> acc{};
        for (std::size_t i = 0; i + 1 < pairs.size(); ++i) {
            const double e_lo =
                -0.5 / (static_cast<double>(pairs[i].n) * pairs[i].n);
            const double e_hi =
                -0.5 / (static_cast<double>(pairs[i + 1].n) * pairs[i + 1].n);
            const std::complex<double> a_lo =
                ses::signed_m_amplitudes(c[pairs[i].idx_cos],
                                         c[pairs[i].idx_sin])
                    .plus;
            const std::complex<double> a_hi =
                ses::signed_m_amplitudes(c[pairs[i + 1].idx_cos],
                                         c[pairs[i + 1].idx_sin])
                    .plus;
            acc += std::conj(a_hi) * a_lo *
                   std::exp(std::complex<double>{0.0, (e_hi - e_lo) * t});
        }
        return acc;
    };
    const std::complex<double> a0 = coherence(0.0);
    EXPECT_GT(std::abs(a0), 0.35);            // angularly localized...
    EXPECT_NEAR(std::arg(a0), 0.0, 1e-12);    // ...pointing at phi = 0
    const double w_kepler = 1.0 / (n_bar * n_bar * n_bar);
    const double t_probe = 0.25 * 3.14159265358979323846 / w_kepler;
    const double dphi = std::arg(coherence(t_probe));  // still < pi: no wrap
    EXPECT_GT(dphi, 0.7 * w_kepler * t_probe);   // CCW, Kepler-rate window
    EXPECT_LT(dphi, 1.8 * w_kepler * t_probe);
}

}  // namespace
