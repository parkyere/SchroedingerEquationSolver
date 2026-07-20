// RED: potential builders (real-valued V on the grid).
// Grid coords are integers -> oracle values exact in binary.


#include <gtest/gtest.h>

#include <cmath>
#include <vector>
import ses.grid;
import ses.potential;

namespace {

using ses::Grid1D;

// coords 0..7 (h = 1).
const Grid1D kGrid{0.0, 8.0, 8};

TEST(AbsorbingMask, OneInInteriorTapersToZeroAtWalls) {
    const ses::Grid1D ax{-8.0, 8.0, 16};  // h = 1: coords -8 .. 7
    const ses::Grid3D g{ax, ax, ax};
    const std::vector<double> m = ses::absorbing_mask(g, 3.0);
    ASSERT_EQ(m.size(), static_cast<std::size_t>(g.size()));
    // coord (0,0,0): deep interior.
    EXPECT_DOUBLE_EQ(m[static_cast<std::size_t>(g.flat(8, 8, 8))], 1.0);
    // coord (-8,-8,-8): on the wall.
    EXPECT_NEAR(m[static_cast<std::size_t>(g.flat(0, 0, 0))], 0.0, 1e-12);
    for (double v : m) {
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0 + 1e-12);
    }
    // coord (-7,0,0): x one step into the 3-wide layer -> sin^2(pi/6); y,z interior.
    const double s = std::sin(0.5 * 3.14159265358979323846 * (1.0 / 3.0));
    EXPECT_NEAR(m[static_cast<std::size_t>(g.flat(1, 8, 8))], s * s, 1e-12);
}

// RED: collapsed axis (n==1) has no walls -> factor 1; the corral open-boundary mask needs this.
TEST(AbsorbingMask, CollapsedAxisHasNoWalls) {
    const ses::Grid1D flat{-1.0, 1.0, 1};
    const std::vector<double> m1 = ses::absorbing_mask(flat, 3.0);
    ASSERT_EQ(m1.size(), 1u);
    EXPECT_DOUBLE_EQ(m1[0], 1.0);

    const ses::Grid1D ax{-16.0, 16.0, 8};  // h = 4: interior well clear of walls
    const ses::Grid3D g{ax, ax, flat};
    const std::vector<double> m = ses::absorbing_mask(g, 4.0);
    // x,y interior; collapsed z -> 1.
    EXPECT_DOUBLE_EQ(m[static_cast<std::size_t>(g.flat(4, 4, 0))], 1.0);
    // x/y wall ramp survives (mask not all-1).
    EXPECT_NEAR(m[static_cast<std::size_t>(g.flat(0, 4, 0))], 0.0, 1e-12);
}

TEST(BarrierPotential, SlabAlongXExactAndZeroElsewhere) {
    const ses::Grid1D ax{-8.0, 8.0, 16};  // h = 1: coords -8 .. 7
    const ses::Grid3D g{ax, ax, ax};
    // slab V=0.25 on x in [0,3), y/z-free.
    const std::vector<double> v = ses::barrier_potential(g, 0.25, 0.0, 3.0);
    ASSERT_EQ(v.size(), static_cast<std::size_t>(g.size()));
    EXPECT_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))], 0.25);    // x = 0
    EXPECT_EQ(v[static_cast<std::size_t>(g.flat(10, 2, 14))], 0.25);  // x = 2, any y/z
    EXPECT_EQ(v[static_cast<std::size_t>(g.flat(7, 8, 8))], 0.0);     // x = -1
    EXPECT_EQ(v[static_cast<std::size_t>(g.flat(11, 8, 8))], 0.0);    // x = 3: half-open
}

TEST(BarrierPotential, OneDimensionalSlabExactAndHalfOpen) {
    // 1D overload (tunneling scene); half-open [x_lo,x_hi) like the 3D slab.
    const ses::Grid1D g{-8.0, 8.0, 16};  // h = 1: coords -8 .. 7
    const std::vector<double> v = ses::barrier_potential(g, 0.25, 0.0, 3.0);
    ASSERT_EQ(v.size(), 16u);
    EXPECT_EQ(v[8], 0.25);   // x = 0
    EXPECT_EQ(v[10], 0.25);  // x = 2
    EXPECT_EQ(v[7], 0.0);    // x = -1
    EXPECT_EQ(v[11], 0.0);   // x = 3: half-open upper edge
}

TEST(AbsorbingMask, OneDimensionalRampMatchesTheAxisFormula) {
    // 1D overload: the per-axis mask factor the 3D mask multiplies.
    const ses::Grid1D g{-8.0, 8.0, 16};  // h = 1: coords -8 .. 7
    const std::vector<double> m = ses::absorbing_mask(g, 3.0);
    ASSERT_EQ(m.size(), 16u);
    EXPECT_DOUBLE_EQ(m[8], 1.0);      // x = 0: deep interior
    EXPECT_NEAR(m[0], 0.0, 1e-12);    // x = -8: on the wall
    const double s = std::sin(0.5 * 3.14159265358979323846 * (1.0 / 3.0));
    EXPECT_NEAR(m[1], s * s, 1e-12);  // x = -7: one step into the layer
    for (double v : m) {
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0 + 1e-12);
    }
}

TEST(DoubleWellPotential, MinimaBarrierAndSymmetryExact) {
    // V(x) = vb ((x/a)^2 - 1)^2.
    const ses::Grid1D g{-8.0, 8.0, 16};  // h = 1: coords -8 .. 7
    const std::vector<double> v = ses::double_well_potential(g, 0.1, 4.0);
    ASSERT_EQ(v.size(), 16u);
    EXPECT_EQ(v[8], 0.1);                 // x = 0: the barrier top
    EXPECT_EQ(v[4], 0.0);                 // x = -4: left minimum
    EXPECT_EQ(v[12], 0.0);                // x = +4: right minimum
    EXPECT_NEAR(v[0], 0.1 * 9.0, 1e-12);  // x = -8: ((4)-1)^2 = 9
    EXPECT_EQ(v[6], v[10]);               // symmetry about the origin
    EXPECT_EQ(v[2], v[14]);
}

TEST(PoschlTellerPotential, DepthAndSechProfileExact) {
    // V(x) = -v0 sech^2((x - x0)/a).
    const ses::Grid1D g{-8.0, 8.0, 16};
    const std::vector<double> v = ses::poschl_teller_potential(g, 0.375, 2.0);
    ASSERT_EQ(v.size(), 16u);
    EXPECT_DOUBLE_EQ(v[8], -0.375);  // x = 0: the well bottom
    const double c1 = std::cosh(1.0);
    EXPECT_DOUBLE_EQ(v[10], -0.375 / (c1 * c1));  // x = 2 = a
    EXPECT_DOUBLE_EQ(v[6], v[10]);                // even about the center
    for (double x : v) {
        EXPECT_LT(x, 0.0);
    }
}

TEST(MorsePotential, MinimumDissociationAndInnerWall) {
    // V(x) = d (1 - e^{-alpha (x - x0)})^2.
    const ses::Grid1D g{-8.0, 8.0, 16};
    const std::vector<double> v = ses::morse_potential(g, 0.3, 0.5, -2.0);
    ASSERT_EQ(v.size(), 16u);
    EXPECT_EQ(v[6], 0.0);  // x = -2 = x0: the minimum
    const double e3 = std::exp(-0.5 * 3.0);  // x = 1: alpha (x - x0) = 1.5
    EXPECT_DOUBLE_EQ(v[9], 0.3 * (1.0 - e3) * (1.0 - e3));
    EXPECT_LT(v[15], 0.3);        // approaches d from below on the right
    EXPECT_GT(v[15], 0.25);
    EXPECT_GT(v[0], 0.3);         // the left wall towers past d
    EXPECT_GT(v[0], 10.0 * v[15]);
}

TEST(HarmonicPotential, ExactValuesAndMinimum) {
    // V(x) = 1/2 omega^2 (x-x0)^2 (convention); omega = 2, x0 = 1.
    const std::vector<double> v = ses::harmonic_potential(kGrid, 2.0, 1.0);
    ASSERT_EQ(v.size(), 8u);
    EXPECT_EQ(v[1], 0.0);   // minimum at the center
    EXPECT_EQ(v[3], 8.0);   // 2 * (3-1)^2
    EXPECT_EQ(v[0], 2.0);   // 2 * (0-1)^2
}

TEST(HarmonicPotential, IsSymmetricAboutCenter) {
    // x0 = 4: V(4+d) == V(4-d).
    const std::vector<double> v = ses::harmonic_potential(kGrid, 1.0, 4.0);
    EXPECT_EQ(v[1], v[7]);
    EXPECT_EQ(v[2], v[6]);
    EXPECT_EQ(v[3], v[5]);
}

TEST(SoftCoulombPotential, ExactValuesAndFiniteAtNucleus) {
    // Z = 1, a = 1, nucleus at x0 = 2.
    const std::vector<double> v = ses::soft_coulomb_potential(kGrid, 1.0, 1.0, 2.0);
    EXPECT_DOUBLE_EQ(v[2], -1.0);                    // center: -Z/a, FINITE
    EXPECT_DOUBLE_EQ(v[4], -1.0 / std::sqrt(5.0));   // dx=2: -1/sqrt(4+1)
    EXPECT_DOUBLE_EQ(v[0], -1.0 / std::sqrt(5.0));   // symmetric partner
}

TEST(SoftCoulombPotential, DeepestAtNucleusAndAttractive) {
    const std::vector<double> v = ses::soft_coulomb_potential(kGrid, 2.0, 0.5, 3.0);
    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_LT(v[i], 0.0);
        EXPECT_GE(v[i], v[3]);       // nowhere deeper than the nucleus
    }
    EXPECT_DOUBLE_EQ(v[3], -4.0);    // -Z/a = -2/0.5
}

// Bare -Z/|r-c| except the nucleus cell (where -Z/r diverges) = analytic cell average -Z*C/h
// (C = integral of 1/r over the unit cube; kCoulombCellAverage; docs/ARCHITECTURE.md). Non-nucleus cells stay exact.
TEST(RegularizedCoulombPotential, ExactAwayFromNucleusAndFiniteAtIt) {
    const ses::Grid1D ax{-8.0, 8.0, 16};  // h = 1, coords -8..7, nucleus point at index 8
    const ses::Grid3D g{ax, ax, ax};
    const std::vector<double> v = ses::regularized_coulomb_potential(g, 1.0, ses::Vec3d{});
    ASSERT_EQ(v.size(), static_cast<std::size_t>(g.size()));
    // nucleus cell (0,0,0): analytic cell average, FINITE.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))],
                     -ses::kCoulombCellAverage);
    // one step along x (r=1): bare -Z/r = -1.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(9, 8, 8))], -1.0);
    // Body-diagonal neighbor (1,1,1): -Z/sqrt(3).
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(9, 9, 9))], -1.0 / std::sqrt(3.0));
    const double center = v[static_cast<std::size_t>(g.flat(8, 8, 8))];
    for (double x : v) {
        EXPECT_LT(x, 0.0);
        EXPECT_TRUE(std::isfinite(x));
        EXPECT_GE(x, center);
    }
}

TEST(RegularizedCoulombPotential, MultiCenterSuperposesWithPerCenterRegularization) {
    // Two protons: each nucleus cell = own analytic average + the other's bare -Z/r (H2+).
    const ses::Grid1D ax{-8.0, 8.0, 16};  // h = 1, coords -8..7
    const ses::Grid3D g{ax, ax, ax};
    const std::vector<ses::Vec3d> centers = {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    const std::vector<double> v =
        ses::regularized_coulomb_potential(g, 1.0, centers);
    ASSERT_EQ(v.size(), static_cast<std::size_t>(g.size()));
    // left nucleus (-1,0,0): own average + other center's -1/2 (r=2).
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(7, 8, 8))],
                     -ses::kCoulombCellAverage - 0.5);
    // midpoint (0,0,0): -1/1 - 1/1 = -2.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))], -2.0);
    // symmetric about the midpoint.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(6, 8, 8))],
                     v[static_cast<std::size_t>(g.flat(10, 8, 8))]);
    for (double x : v) {
        EXPECT_LT(x, 0.0);
        EXPECT_TRUE(std::isfinite(x));
    }
}

TEST(SnapToGrid, RoundsEachAxisToTheNearestGridPointAndClamps) {
    // Multi-center Coulomb regularizes only exact-hit cells, so centers must snap to
    // lattice points (off-grid -> arbitrary -Z/r depth). h=1, coords -8..7; xmax not a
    // point (periodic) -> 7.7 clamps to 7.
    const ses::Grid1D ax{-8.0, 8.0, 16};
    const ses::Grid3D g{ax, ax, ax};
    const ses::Vec3d s = ses::snap_to_grid(g, {0.4, -0.6, 7.7});
    EXPECT_DOUBLE_EQ(s.x, 0.0);
    EXPECT_DOUBLE_EQ(s.y, -1.0);
    EXPECT_DOUBLE_EQ(s.z, 7.0);
    const ses::Vec3d lo = ses::snap_to_grid(g, {-9.3, -8.4, 0.2});
    EXPECT_DOUBLE_EQ(lo.x, -8.0);
    EXPECT_DOUBLE_EQ(lo.y, -8.0);
    EXPECT_DOUBLE_EQ(lo.z, 0.0);
}

TEST(RegularizedCoulombPotential, NucleusCellScalesWithChargeAndInverseSpacing) {
    // cell average -Z*C/h: linear in Z and 1/h.
    const ses::Grid1D ax{-4.0, 4.0, 16};  // h = 0.5, nucleus point at index 8
    const ses::Grid3D g{ax, ax, ax};
    const std::vector<double> v = ses::regularized_coulomb_potential(g, 2.0, ses::Vec3d{});
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))],
                     -2.0 * ses::kCoulombCellAverage / 0.5);
    // away from nucleus: exact -Z/r (r=0.5, one step along x).
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(9, 8, 8))], -2.0 / 0.5);
}

}  // namespace
