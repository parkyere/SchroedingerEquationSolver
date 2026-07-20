// RED: 1D periodic lattice (Bloch) core.
//
// V(x) = V0 sin^2(kL x): smooth keeps the FFT split-operator spectral
// (user's numerics call; Kronig-Penney kinks would Gibbs-ring). a = pi/kL,
// reciprocal G = 2 kL.
//
// Tilted propagator: force F enters as comoving gauge A(t) = -F t, kinetic
// phase rebuilt each step with the midpoint A (exact for linear A -> round-off).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

import ses.bloch;
import ses.field;
import ses.grid;
import ses.wavepacket;

namespace {

TEST(LatticeBands, FreeLimitFoldsTheParabola) {
    const double kl = 1.0;
    const double g2 = 2.0 * kl;
    for (const double q : {0.0, 0.3, 0.9}) {
        const std::vector<double> e = ses::lattice_bands(0.0, kl, q, 4);
        ASSERT_EQ(e.size(), 4u);
        std::vector<double> want;
        for (int m = -3; m <= 3; ++m) {
            const double k = q + m * g2;
            want.push_back(0.5 * k * k);
        }
        std::sort(want.begin(), want.end());
        for (int n = 0; n < 4; ++n) {
            EXPECT_NEAR(e[static_cast<std::size_t>(n)],
                        want[static_cast<std::size_t>(n)], 1e-9)
                << "band " << n << " at q = " << q;
        }
    }
}

TEST(LatticeBands, FirstGapAtTheZoneEdgeIsHalfV0) {
    const double kl = 1.0;
    const double v0 = 0.1;  // weak lattice: first-order PT regime
    const std::vector<double> e = ses::lattice_bands(v0, kl, kl, 2);
    EXPECT_NEAR(e[1] - e[0], 0.5 * v0, 0.1 * 0.5 * v0);
    // bands stay monotone across the half zone (no crossing)
    const std::vector<double> mid = ses::lattice_bands(v0, kl, 0.5 * kl, 2);
    EXPECT_LT(mid[0], e[0]);
    EXPECT_GT(mid[1], e[1]);
}

TEST(TiltedSplitOperator1D, FreeTiltIsExactlyUniformAcceleration) {
    const ses::Grid1D g{-40.0, 40.0, 2048};
    const std::vector<double> zero(static_cast<std::size_t>(g.n), 0.0);
    const double f = 0.3;
    const double dt = 0.005;
    ses::TiltedSplitOperator1D prop{g, zero, dt, f};
    const double k0 = 1.0;
    const double x0 = -20.0;
    ses::Field1D psi = ses::gaussian_wavepacket(g, x0, 3.0, k0);
    const int steps = 1600;
    prop.step(psi, steps);
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < g.n; ++i) {
        const double w = std::norm(psi[i]);
        num += g.coord(i) * w;
        den += w;
    }
    const double t = steps * dt;
    EXPECT_NEAR(num / den, x0 + k0 * t + 0.5 * f * t * t, 1e-6);
    EXPECT_NEAR(prop.drift(), f * t, 1e-12);
}

TEST(TiltedSplitOperator1D, BlochOscillationReturnsInsteadOfRunningAway) {
    // depth s = V0/E_R = 3: open gaps, negligible Zener (F a << gap^2/bandwidth)
    const double kl = 1.0;
    const double a = std::numbers::pi / kl;
    const ses::Grid1D g{-13.0 * a, 13.0 * a, 2048};
    const double v0 = 1.5;
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double s = std::sin(kl * g.coord(i));
        v[static_cast<std::size_t>(i)] = v0 * s * s;
    }
    const double f = 0.05;
    const double t_b = 2.0 * kl / f;  // Bloch period G/F
    const double dt = 0.01;
    ses::TiltedSplitOperator1D prop{g, v, dt, f};
    // broad packet at a well minimum: ground band, q ~ 0
    ses::Field1D psi = ses::gaussian_wavepacket(g, 0.0, 6.0, 0.0);
    auto mean_x = [&] {
        double num = 0.0;
        double den = 0.0;
        for (int i = 0; i < g.n; ++i) {
            const double w = std::norm(psi[i]);
            num += g.coord(i) * w;
            den += w;
        }
        return num / den;
    };
    const double x0 = mean_x();
    const int steps = static_cast<int>(t_b / dt + 0.5);
    double excursion = 0.0;
    for (int s = 0; s < steps; s += 50) {
        prop.step(psi, std::min(50, steps - s));
        excursion = std::max(excursion, std::abs(mean_x() - x0));
    }
    const double free_fall = 0.5 * f * t_b * t_b;
    EXPECT_LT(excursion, 0.15 * free_fall);
    EXPECT_LT(std::abs(mean_x() - x0), 1.5);
}

}  // namespace
