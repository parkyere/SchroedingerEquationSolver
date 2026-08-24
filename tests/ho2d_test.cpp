// RED: 2D circular HO ladder -- a-dag adds one omega; a|0> = 0; up-down round-trips.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

import ses.lattice2d;
import ses.field;
import ses.grid;
import ses.observables;
import ses.potential;

namespace {

ses::Field3D ho_ground(const ses::Grid3D& g, double omega) {
    ses::Field3D psi{g};
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            psi(i, j, 0) = std::exp(-0.5 * omega * (x * x + y * y));
        }
    }
    ses::normalize(psi);
    return psi;
}

TEST(Ho2dLadder, RaisesExactlyOneOmegaAndRoundTrips) {
    const double omega = 0.5;
    const ses::Grid3D g{ses::Grid1D{-14.0, 14.0, 128},
                        ses::Grid1D{-14.0, 14.0, 128},
                        ses::Grid1D{0.0, 2.0, 1}};
    // z axis holds the single z = 0 plane, so the 3D builder matches the 2D slab.
    const std::vector<double> v =
        ses::harmonic_potential(g, omega, ses::Vec3d{});
    const ses::Field3D ground = ho_ground(g, omega);
    const double e0 = ses::mean_energy(ground, v);
    EXPECT_NEAR(e0, omega, 0.01 * omega);  // E_00 = omega (2D zero point)

    ses::Field3D up = ses::ho2d_ladder(ground, omega, ses::Rung::Raise);
    const double n_up = ses::norm_sq(up);
    ASSERT_GT(n_up, 0.5);
    ses::normalize(up);
    const double e1 = ses::mean_energy(up, v);
    EXPECT_NEAR(e1 - e0, omega, 0.02 * omega);

    ses::Field3D down = ses::ho2d_ladder(up, omega, ses::Rung::Lower);
    ses::normalize(down);
    EXPECT_GT(std::norm(ses::inner_product(down, ground)), 0.999);

    ses::Field3D dead = ses::ho2d_ladder(ground, omega, ses::Rung::Lower);
    EXPECT_LT(ses::norm_sq(dead), 1e-3);  // a|0> = 0
}

}  // namespace
