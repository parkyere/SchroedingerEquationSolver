// RED: potential builders (real-valued V on the grid).
// Grid coords are integers -> oracle values exact in binary.


#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numbers>
#include <utility>
#include <vector>
import ses.grid;
import ses.potential;
import ses.field;
import ses.imaginary_time;
import ses.observables;
import ses.wavepacket;

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
    const double s = std::sin(0.5 * std::numbers::pi * (1.0 / 3.0));
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
    const double s = std::sin(0.5 * std::numbers::pi * (1.0 / 3.0));
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

// Finite-volume bare Coulomb: every cell within kCoulombAverageRadius cells
// of a nucleus takes the CUBE AVERAGE of -Z/|r-c| (closed form, exact for the
// nucleus cell too); beyond that the point value -Z/r (h^4-close to the
// average). On-grid nuclei keep -Z*C/h (C = kCoulombCellAverage).
TEST(RegularizedCoulombPotential, CubeAveragedNearTheNucleusExactBeyond) {
    const ses::Grid1D ax{-8.0, 8.0, 16};  // h = 1, coords -8..7, nucleus point at index 8
    const ses::Grid3D g{ax, ax, ax};
    const std::vector<double> v = ses::regularized_coulomb_potential(g, 1.0, ses::Vec3d{});
    ASSERT_EQ(v.size(), static_cast<std::size_t>(g.size()));
    // nucleus cell (0,0,0): analytic cell average, FINITE.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))],
                     -ses::kCoulombCellAverage);
    // one step along x (r=1): the cube average, not -1.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(9, 8, 8))],
                     -ses::coulomb_cube_average(ses::Vec3d{1.0, 0.0, 0.0}, 1.0));
    // Body-diagonal neighbor (1,1,1).
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(9, 9, 9))],
                     -ses::coulomb_cube_average(ses::Vec3d{1.0, 1.0, 1.0}, 1.0));
    // (2,2,0): r = 2.83 < 3 cells -> still averaged.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(10, 10, 8))],
                     -ses::coulomb_cube_average(ses::Vec3d{2.0, 2.0, 0.0}, 1.0));
    // r = 3 cells is OUTSIDE the averaged ball (strict): bare -1/3; r = 4: -1/4.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(11, 8, 8))], -1.0 / 3.0);
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(12, 8, 8))], -0.25);
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
    // left nucleus (-1,0,0): own average + the other center's cube average at r=2.
    const double a1 = ses::coulomb_cube_average(ses::Vec3d{1.0, 0.0, 0.0}, 1.0);
    const double a2 = ses::coulomb_cube_average(ses::Vec3d{2.0, 0.0, 0.0}, 1.0);
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(7, 8, 8))],
                     -ses::kCoulombCellAverage - a2);
    // midpoint (0,0,0): two averaged r=1 contributions.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))], -2.0 * a1);
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

// Static-E tilt: V += e0 * coord along the chosen axis, other axes untouched.
TEST(TiltedPotential, AddsTheAxisTiltAndNothingElse) {
    const ses::Grid1D ax{-4.0, 4.0, 8};
    const ses::Grid3D g{ax, ax, ax};
    const std::vector<double> base(static_cast<std::size_t>(g.size()), 1.5);
    for (int axis = 0; axis < 3; ++axis) {
        const std::vector<double> v =
            ses::tilted_potential(base, g, 0.02, axis);
        for (const int i : {0, 3, 7}) {
            const double c = axis == 0   ? g.x.coord(i)
                             : axis == 1 ? g.y.coord(i)
                                         : g.z.coord(i);
            const int ix = axis == 0 ? i : 2;
            const int iy = axis == 1 ? i : 2;
            const int iz = axis == 2 ? i : 2;
            EXPECT_DOUBLE_EQ(
                v[static_cast<std::size_t>(g.flat(ix, iy, iz))],
                1.5 + 0.02 * c)
                << "axis " << axis << " i " << i;
        }
        // Moving along a DIFFERENT axis leaves the tilt unchanged.
        const int other = (axis + 1) % 3;
        const int a0[3] = {1, 1, 1};
        int a1[3] = {1, 1, 1};
        a1[other] = 5;
        EXPECT_DOUBLE_EQ(
            v[static_cast<std::size_t>(g.flat(a0[0], a0[1], a0[2]))],
            v[static_cast<std::size_t>(g.flat(a1[0], a1[1], a1[2]))]);
    }
}

TEST(TiltedPotential, OddInTheFieldSign) {
    const ses::Grid1D ax{-4.0, 4.0, 8};
    const ses::Grid3D g{ax, ax, ax};
    const std::vector<double> base(static_cast<std::size_t>(g.size()), 0.0);
    const auto vp = ses::tilted_potential(base, g, 0.5, 2);
    const auto vm = ses::tilted_potential(base, g, -0.5, 2);
    const std::size_t p = static_cast<std::size_t>(g.flat(2, 2, 6));
    EXPECT_DOUBLE_EQ(vp[p], -vm[p]);
    EXPECT_NE(vp[p], 0.0);  // z-coord at index 6 is nonzero
}

TEST(RegularizedCoulombPotential, NucleusCellScalesWithChargeAndInverseSpacing) {
    // cell average -Z*C/h: linear in Z and 1/h.
    const ses::Grid1D ax{-4.0, 4.0, 16};  // h = 0.5, nucleus point at index 8
    const ses::Grid3D g{ax, ax, ax};
    const std::vector<double> v = ses::regularized_coulomb_potential(g, 2.0, ses::Vec3d{});
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))],
                     -2.0 * ses::kCoulombCellAverage / 0.5);
    // one step along x (r = h): Z times the cube average at that spacing.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(9, 8, 8))],
                     -2.0 * ses::coulomb_cube_average(ses::Vec3d{0.5, 0.0, 0.0}, 0.5));
    // 2 cells out (r = 1.0 = 2h): still averaged; 4 cells out: bare -Z/r.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(10, 8, 8))],
                     -2.0 * ses::coulomb_cube_average(ses::Vec3d{1.0, 0.0, 0.0}, 0.5));
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(12, 8, 8))], -2.0 / 2.0);
}

// A MOVING nucleus (rigid rotor) sweeps past grid points: -Z/r at r -> 0
// would blow the Trotter budget, and a fixed cap plus point-sampled
// neighbours modulates <V> with the sub-cell landing (egg-box). Each near
// cell takes the exact cube average of -Z/|r-c| for the nucleus where it IS.
TEST(RegularizedCoulomb, AveragesEachCellAroundAnOffGridNucleus) {
    const ses::Grid1D axis{-4.0, 4.0, 16};  // h = 0.5
    const ses::Grid3D g{axis, axis, axis};
    const double h = 0.5;
    const ses::Vec3d c{0.1, 0.0, 0.0};
    const std::vector<double> v = ses::regularized_coulomb_potential(g, 1.0, c);
    // origin cell holds the nucleus 0.1 off its center: its own average.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))],
                     -ses::coulomb_cube_average(ses::Vec3d{-0.1, 0.0, 0.0}, h));
    EXPECT_GT(v[static_cast<std::size_t>(g.flat(8, 8, 8))],
              -ses::kCoulombCellAverage / h);  // shallower than on-center
    // next point (0.5, 0, 0): 0.4 from the nucleus -> that cell's average.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(9, 8, 8))],
                     -ses::coulomb_cube_average(ses::Vec3d{0.4, 0.0, 0.0}, h));
    // Sampling symmetry: the cube average is even in d.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(7, 8, 8))],
                     -ses::coulomb_cube_average(ses::Vec3d{0.6, 0.0, 0.0}, h));
}

// ---- coulomb_cube_average(d, h) = (1/h^3) integral over the cube of side h
// centered at the origin of 1/|r - d|: the potential of a homogeneous cube
// (Waldvogel 1976), closed form valid for d inside the cube (the nucleus cell)
// as well as outside. Oracles: symmetry constants, Poisson/Laplace, high-order
// Gauss-Legendre at regular points, the far-field point value.

// n-point Gauss-Legendre nodes/weights on [-1, 1] (Newton on P_n).
std::vector<std::pair<double, double>> gauss_legendre(int n) {
    auto legendre = [n](double x) {
        double p0 = 1.0;
        double p1 = x;
        for (int k = 2; k <= n; ++k) {
            const double p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
            p0 = p1;
            p1 = p2;
        }
        const double dp = n * (x * p1 - p0) / (x * x - 1.0);
        return std::pair<double, double>{p1, dp};
    };
    std::vector<std::pair<double, double>> nw;
    for (int i = 0; i < n; ++i) {
        double x = std::cos(std::numbers::pi * (i + 0.75) / (n + 0.5));
        for (int it = 0; it < 50; ++it) {
            const auto [p, dp] = legendre(x);
            const double dx = p / dp;
            x -= dx;
            if (std::abs(dx) < 1e-15) {
                break;
            }
        }
        const double dp = legendre(x).second;
        nw.emplace_back(x, 2.0 / ((1.0 - x * x) * dp * dp));
    }
    return nw;
}

// Product Gauss-Legendre over the cube; converges geometrically while the
// singularity stays off the cube (regular points only).
double cube_average_quadrature(ses::Vec3d d, double h, int n) {
    const auto nw = gauss_legendre(n);
    double acc = 0.0;
    for (const auto& [x, wx] : nw) {
        for (const auto& [y, wy] : nw) {
            for (const auto& [z, wz] : nw) {
                const double px = 0.5 * h * x - d.x;
                const double py = 0.5 * h * y - d.y;
                const double pz = 0.5 * h * z - d.z;
                acc += wx * wy * wz / std::sqrt(px * px + py * py + pz * pz);
            }
        }
    }
    return acc / 8.0;  // (h/2)^3 Jacobian over h^3
}

TEST(CoulombCubeAverage, CenterIsTheTabulatedConstantOverH) {
    for (const double h : {1.0, 0.3125, 0.05}) {
        EXPECT_NEAR(ses::coulomb_cube_average(ses::Vec3d{}, h) * h,
                    ses::kCoulombCellAverage, 2e-7)
            << "h " << h;
    }
}

TEST(CoulombCubeAverage, CornerIsHalfTheCenterValue) {
    // A corner of the h cube is the center of a 2h cube's octant:
    // (2h)^2 C / 8 / h^3 = C / (2h). Every sign choice is the same corner.
    const double h = 0.7;
    for (const double sx : {-0.5, 0.5}) {
        for (const double sy : {-0.5, 0.5}) {
            for (const double sz : {-0.5, 0.5}) {
                const ses::Vec3d corner{sx * h, sy * h, sz * h};
                EXPECT_NEAR(ses::coulomb_cube_average(corner, h) * h,
                            0.5 * ses::kCoulombCellAverage, 2e-7)
                    << sx << " " << sy << " " << sz;
            }
        }
    }
}

TEST(CoulombCubeAverage, MatchesHighOrderQuadratureAtRegularPoints) {
    const double h = 0.3125;
    const ses::Vec3d pts[] = {{h, 0.0, 0.0},
                              {1.3 * h, 0.4 * h, -0.7 * h},
                              {0.0, 2.0 * h, 0.5 * h},
                              {-1.5 * h, -1.5 * h, 1.5 * h}};
    for (const ses::Vec3d& d : pts) {
        const double want = cube_average_quadrature(d, h, 20);
        const double got = ses::coulomb_cube_average(d, h);
        EXPECT_NEAR(got, want, 1e-9 * want) << d.x << " " << d.y << " " << d.z;
    }
}

TEST(CoulombCubeAverage, NearestNeighbourCarriesTheFourthOrderCorrection) {
    // Cube-average Taylor expansion of 1/r about (h,0,0): the Laplacian term
    // vanishes (harmonic), the h^4 terms give (24+9+9)/1920 - (12+12-3)/576
    // = -0.01458 relative -> the neighbour is ~1.5% shallower than -1/h.
    const double h = 0.3125;
    const double a = ses::coulomb_cube_average(ses::Vec3d{h, 0.0, 0.0}, h) * h;
    EXPECT_NEAR(a, 1.0 - 0.01458, 1e-3);
}

TEST(CoulombCubeAverage, SatisfiesPoissonInsideAndLaplaceOutside) {
    // Unit-density cube: laplacian(avg) = -4 pi / h^3 inside, 0 outside.
    const double h = 0.5;
    const double eps = 1e-3 * h;
    auto laplacian = [&](ses::Vec3d d) {
        double acc = 0.0;
        for (int a = 0; a < 3; ++a) {
            ses::Vec3d dp = d;
            ses::Vec3d dm = d;
            (&dp.x)[a] += eps;
            (&dm.x)[a] -= eps;
            acc += ses::coulomb_cube_average(dp, h) +
                   ses::coulomb_cube_average(dm, h) -
                   2.0 * ses::coulomb_cube_average(d, h);
        }
        return acc / (eps * eps);
    };
    const double rho = 4.0 * std::numbers::pi / (h * h * h);
    EXPECT_NEAR(laplacian(ses::Vec3d{0.1 * h, 0.05 * h, -0.2 * h}), -rho,
                1e-5 * rho);
    EXPECT_NEAR(laplacian(ses::Vec3d{0.35 * h, -0.4 * h, 0.3 * h}), -rho,
                1e-5 * rho);
    EXPECT_NEAR(laplacian(ses::Vec3d{1.3 * h, 0.4 * h, -0.7 * h}), 0.0,
                1e-5 * rho);
    EXPECT_NEAR(laplacian(ses::Vec3d{0.0, 0.0, 2.0 * h}), 0.0, 1e-5 * rho);
}

TEST(CoulombCubeAverage, FiniteAndContinuousOnFacesEdgesAndCorners) {
    // The closed form's logs/arctans degenerate when a corner offset has a
    // zero coordinate (d on a face plane, edge line, or corner): every such
    // point must be finite and equal the limit from a hair away.
    const double h = 0.25;
    const double tiny = 1e-7 * h;
    for (const double sx : {-0.5, 0.0, 0.5}) {
        for (const double sy : {-0.5, 0.0, 0.5}) {
            for (const double sz : {-0.5, 0.0, 0.5}) {
                const ses::Vec3d d{sx * h, sy * h, sz * h};
                const double a = ses::coulomb_cube_average(d, h);
                ASSERT_TRUE(std::isfinite(a)) << sx << " " << sy << " " << sz;
                EXPECT_GT(a, 0.0);
                const double b = ses::coulomb_cube_average(
                    ses::Vec3d{d.x + tiny, d.y + tiny, d.z + tiny}, h);
                EXPECT_NEAR(a, b, 1e-6 * a) << sx << " " << sy << " " << sz;
            }
        }
    }
    // On the face plane but outside the cube's shadow: also degenerate.
    const ses::Vec3d off{0.5 * h, 2.0 * h, 0.0};
    EXPECT_NEAR(ses::coulomb_cube_average(off, h),
                cube_average_quadrature(off, h, 20), 1e-9);
}

TEST(CoulombCubeAverage, FarFieldIsThePointValue) {
    const double h = 0.3125;
    EXPECT_NEAR(ses::coulomb_cube_average(ses::Vec3d{10.0 * h, 0.0, 0.0}, h) *
                    10.0 * h,
                1.0, 1e-5);
    // At the averaging cutoff (3 cells) the h^4 term is ~2e-4: the switch to
    // the point value beyond it is that smooth.
    EXPECT_NEAR(ses::coulomb_cube_average(ses::Vec3d{3.0 * h, 0.0, 0.0}, h) *
                    3.0 * h,
                1.0, 5e-4);
}

// ---- egg-box: the sampled Coulomb energy must not depend on where the
// nucleus sits inside a cell (rigid-rotor nuclei sweep the lattice; a
// modulation is a spurious torque and a J drift). h = 0.3125 = the 256^3/+-40
// scene spacing; 32^3 keeps the radix-2 FFT and holds 1s (e^-10 at the wall).

const ses::Grid3D kEggBoxGrid{ses::Grid1D{-5.0, 5.0, 32},
                              ses::Grid1D{-5.0, 5.0, 32},
                              ses::Grid1D{-5.0, 5.0, 32}};
constexpr double kEggBoxH = 0.3125;

// Sub-cell offsets: on-point, axis half-cell, face/edge/corner-ish, generic.
const std::vector<ses::Vec3d> kEggBoxOffsets = {
    {0.0, 0.0, 0.0},
    {0.25 * kEggBoxH, 0.0, 0.0},
    {0.5 * kEggBoxH, 0.0, 0.0},
    {0.5 * kEggBoxH, 0.5 * kEggBoxH, 0.0},
    {0.5 * kEggBoxH, 0.5 * kEggBoxH, 0.5 * kEggBoxH},
    {0.37 * kEggBoxH, 0.11 * kEggBoxH, 0.23 * kEggBoxH},
};

double spread(const std::vector<double>& e) {
    return *std::max_element(e.begin(), e.end()) -
           *std::min_element(e.begin(), e.end());
}

// <V> of the frozen exact 1s cusp e^-|r-c| sampled on the lattice (grid
// normalized): isolates the potential builder from the kinetic operator.
TEST(RegularizedCoulomb, EggBoxOfTheFrozenOneSCloudIsBelowBudget) {
    const ses::Grid3D& g = kEggBoxGrid;
    std::vector<double> e;
    for (const ses::Vec3d& c : kEggBoxOffsets) {
        const std::vector<double> v =
            ses::regularized_coulomb_potential(g, 1.0, c);
        double num = 0.0;
        double den = 0.0;
        ses::for_each_cell(g, [&](int i, int j, int k, int flat) {
            const double dx = g.x.coord(i) - c.x;
            const double dy = g.y.coord(j) - c.y;
            const double dz = g.z.coord(k) - c.z;
            const double w = std::exp(-2.0 * std::sqrt(dx * dx + dy * dy + dz * dz));
            num += w * v[static_cast<std::size_t>(flat)];
            den += w;
        });
        e.push_back(num / den);
        std::printf("  frozen 1s  c = (%.3f, %.3f, %.3f) h  <V> = %.6f Ha\n",
                    c.x / kEggBoxH, c.y / kEggBoxH, c.z / kEggBoxH, e.back());
    }
    const double dv = spread(e);
    std::printf("  frozen 1s egg-box spread = %.3e Ha\n", dv);
    EXPECT_LT(dv, 1.0e-3);
}

// The discrete ground state itself (ITP, kinetic + potential): the energy a
// rotor scene's <H_el> reads must be translation invariant to the budget.
TEST(RegularizedCoulomb, EggBoxOfTheRelaxedGroundStateIsBelowBudget) {
    const ses::Grid3D& g = kEggBoxGrid;
    std::vector<double> e;
    for (const ses::Vec3d& c : kEggBoxOffsets) {
        const std::vector<double> v =
            ses::regularized_coulomb_potential(g, 1.0, c);
        ses::Field3D psi = ses::gaussian_wavepacket(
            g, c, ses::Vec3d{1.0, 1.0, 1.0}, ses::Vec3d{});
        const ses::ImaginaryTimePropagator3D relaxer{g, v, 0.05};
        relaxer.relax(psi, 400);  // tau = 20: 2s admixture e^-15
        e.push_back(ses::mean_energy(psi, v));
        std::printf("  relaxed 1s c = (%.3f, %.3f, %.3f) h  E0 = %.6f Ha\n",
                    c.x / kEggBoxH, c.y / kEggBoxH, c.z / kEggBoxH, e.back());
    }
    const double de = spread(e);
    std::printf("  relaxed 1s egg-box spread = %.3e Ha\n", de);
    EXPECT_LT(de, 1.0e-3);
}

}  // namespace
