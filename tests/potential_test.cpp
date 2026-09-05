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

// Nyquist band-limited bare Coulomb: V_i = -Z (2/pi) Si(pi r/h) / r at EVERY
// cell (no window: the Gibbs tail ~cos(pi r/h)/(pi r^2/h) is what the grid
// aliases), V(0) = -2Z/h. Parameter-free (K = pi/h is the grid's own band),
// deeper than -Z/r at the nearest cell (-1.18 Z/h), 10 mHa-accurate 1s at
// h = 0.31 and a 14x smaller egg-box than the cube average
// (docs/ARCHITECTURE.md: why not soft-Coulomb).
TEST(RegularizedCoulombPotential, IsTheBandLimitedCoulombEverywhere) {
    const ses::Grid1D ax{-8.0, 8.0, 16};  // h = 1, coords -8..7, nucleus at index 8
    const ses::Grid3D g{ax, ax, ax};
    const std::vector<double> v = ses::regularized_coulomb_potential(g, 1.0, ses::Vec3d{});
    ASSERT_EQ(v.size(), static_cast<std::size_t>(g.size()));
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))], -2.0);  // -2Z/h
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(9, 8, 8))],
                     -ses::band_limited_coulomb(1.0, 1.0));
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(9, 9, 9))],
                     -ses::band_limited_coulomb(std::sqrt(3.0), 1.0));
    // No window: 7 cells out is still the band-limited value, not -1/7.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(15, 8, 8))],
                     -ses::band_limited_coulomb(7.0, 1.0));
    EXPECT_NE(v[static_cast<std::size_t>(g.flat(15, 8, 8))], -1.0 / 7.0);
    const double center = v[static_cast<std::size_t>(g.flat(8, 8, 8))];
    for (double x : v) {
        EXPECT_LT(x, 0.0);
        EXPECT_TRUE(std::isfinite(x));
        EXPECT_GE(x, center);
    }
}

TEST(RegularizedCoulombPotential, MultiCenterSuperposesWithPerCenterRegularization) {
    const ses::Grid1D ax{-8.0, 8.0, 16};  // h = 1
    const ses::Grid3D g{ax, ax, ax};
    const std::vector<ses::Vec3d> centers = {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    const std::vector<double> v = ses::regularized_coulomb_potential(g, 1.0, centers);
    ASSERT_EQ(v.size(), static_cast<std::size_t>(g.size()));
    const double b1 = ses::band_limited_coulomb(1.0, 1.0);
    const double b2 = ses::band_limited_coulomb(2.0, 1.0);
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(7, 8, 8))], -2.0 - b2);
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))], -2.0 * b1);
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(6, 8, 8))],
                     v[static_cast<std::size_t>(g.flat(10, 8, 8))]);
}

TEST(RegularizedCoulombPotential, ScalesWithChargeAndInverseSpacing) {
    const ses::Grid1D ax{-4.0, 4.0, 16};  // h = 0.5
    const ses::Grid3D g{ax, ax, ax};
    const std::vector<double> v = ses::regularized_coulomb_potential(g, 2.0, ses::Vec3d{});
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))], -2.0 * 2.0 / 0.5);
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(9, 8, 8))],
                     -2.0 * ses::band_limited_coulomb(0.5, 0.5));
}

TEST(RegularizedCoulomb, AnOffGridNucleusIsTheSameFunctionOfDistance) {
    const ses::Grid1D axis{-4.0, 4.0, 16};  // h = 0.5
    const ses::Grid3D g{axis, axis, axis};
    const ses::Vec3d c{0.1, 0.0, 0.0};
    const std::vector<double> v = ses::regularized_coulomb_potential(g, 1.0, c);
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))],
                     -ses::band_limited_coulomb(0.1, 0.5));
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(9, 8, 8))],
                     -ses::band_limited_coulomb(0.4, 0.5));
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(7, 8, 8))],
                     -ses::band_limited_coulomb(0.6, 0.5));
}

// ---- Si(x) = integral_0^x sin t / t dt: series below 4, the Abramowitz-Stegun
// 5.2.38/39 rational f, g above (|err| < 5e-7). Oracle: composite Simpson.
double si_simpson(double x) {
    const int n = 200000;  // even
    const double h = x / n;
    double acc = 1.0;  // t = 0
    for (int i = 1; i < n; ++i) {
        const double t = i * h;
        acc += (i % 2 == 1 ? 4.0 : 2.0) * std::sin(t) / t;
    }
    acc += std::sin(x) / x;
    return acc * h / 3.0;
}

TEST(SineIntegral, MatchesQuadratureToHalfAMicro) {
    for (const double x : {0.1, 0.5, 1.0, 2.0, 3.9, 4.0, 4.1, 5.0, 7.0, 10.0,
                           15.0, 30.0, 50.0, 120.0, 300.0}) {
        EXPECT_NEAR(ses::sine_integral(x), si_simpson(x), 6e-7) << "x " << x;
    }
}

TEST(SineIntegral, IsOddZeroAtZeroAndApproachesPiOverTwo) {
    EXPECT_DOUBLE_EQ(ses::sine_integral(0.0), 0.0);
    EXPECT_DOUBLE_EQ(ses::sine_integral(-2.5), -ses::sine_integral(2.5));
    EXPECT_NEAR(ses::sine_integral(1.0e4), 0.5 * std::numbers::pi, 2e-4);
    EXPECT_NEAR(ses::sine_integral(std::numbers::pi), 1.851937051982466, 1e-9);
}

TEST(BandLimitedCoulomb, OriginLimitNearestCellAndFarTail) {
    const double h = 0.3125;
    // r -> 0: (2/pi) K = 2/h, continuous.
    EXPECT_NEAR(ses::band_limited_coulomb(1e-9, h) * h, 2.0, 1e-8);
    // The nearest cell is 18% deeper than 1/h: (2/pi) Si(pi) = 1.179.
    EXPECT_NEAR(ses::band_limited_coulomb(h, h) * h, 1.17898, 1e-4);
    // Gibbs tail ~ cos(pi r/h) / (pi r^2/h): 0.7% at 30 cells, 0.07% at 300.
    EXPECT_NEAR(ses::band_limited_coulomb(30.0 * h, h) * 30.0 * h, 1.0, 1e-2);
    EXPECT_NEAR(ses::band_limited_coulomb(300.0 * h, h) * 300.0 * h, 1.0, 1e-3);
    EXPECT_GT(std::abs(ses::band_limited_coulomb(30.0 * h, h) * 30.0 * h - 1.0),
              1e-4);  // it IS a ripple, not bare 1/r
}

TEST(SnapToGrid, RoundsEachAxisToTheNearestGridPointAndClamps) {
    // Static molecular centers snap to lattice points so their nucleus cells keep
    // the tabulated on-point average. h=1, coords -8..7; xmax not a point
    // (periodic) -> 7.7 clamps to 7.
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

// ---- egg-box: the sampled Coulomb energy must not depend on where the
// nucleus sits inside a cell (rigid-rotor nuclei sweep the lattice; a
// modulation is a spurious torque and a J drift). h = 0.3125 = the 256^3/+-40
// scene spacing; 32^3 keeps the radix-2 FFT and holds 1s (e^-10 at the wall).
// Measured spreads (point-sampled -> cube-averaged -> band-limited builder):
//   smooth e^-r^2 cloud, frozen:  13.1 -> 1.00 -> 0.000 mHa
//   relaxed ground state, h:      15.8 -> 3.84 -> 0.27 mHa
//   relaxed ground state, h/2:    3.66 -> 0.42 -> 0.026 mHa
//   E(1s), h / h/2:  -0.4923/-  -> -0.4913/-0.4970 -> -0.4996/-0.5007 (exact -0.5)
// The band limit kills the aliasing outright; the residual is the density's.

const ses::Grid3D kEggBoxGrid{ses::Grid1D{-5.0, 5.0, 32},
                              ses::Grid1D{-5.0, 5.0, 32},
                              ses::Grid1D{-5.0, 5.0, 32}};
constexpr double kEggBoxH = 0.3125;

// Sub-cell offsets: on-point, axis quarter/half-cell, edge, corner, generic.
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

// ITP ground-state energy of hydrogen with the nucleus at c.
double relaxed_ground_energy(const ses::Grid3D& g, ses::Vec3d c, double dtau,
                             int steps) {
    const std::vector<double> v = ses::regularized_coulomb_potential(g, 1.0, c);
    ses::Field3D psi =
        ses::gaussian_wavepacket(g, c, ses::Vec3d{1.0, 1.0, 1.0}, ses::Vec3d{});
    const ses::ImaginaryTimePropagator3D relaxer{g, v, dtau};
    relaxer.relax(psi, steps);
    return ses::mean_energy(psi, v);
}

// <V> of a frozen SMOOTH cloud e^-|r-c|^2 (grid normalized): no cusp, so the
// modulation is the builder's own aliasing, nothing else.
TEST(RegularizedCoulomb, EggBoxOfASmoothCloudIsBelowBudget) {
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
            const double w = std::exp(-(dx * dx + dy * dy + dz * dz));
            num += w * v[static_cast<std::size_t>(flat)];
            den += w;
        });
        e.push_back(num / den);
        std::printf("  smooth cloud c = (%.3f, %.3f, %.3f) h  <V> = %.6f Ha\n",
                    c.x / kEggBoxH, c.y / kEggBoxH, c.z / kEggBoxH, e.back());
    }
    const double dv = spread(e);
    std::printf("  smooth cloud egg-box spread = %.3e Ha\n", dv);
    EXPECT_LT(dv, 5.0e-5);
}

// The discrete ground state itself (ITP, kinetic + potential): the energy a
// rotor scene's <H_el> reads. Point 15.8, cube average 3.8, band limit 0.27.
// The spread is a property of the discrete H, not of the ITP's Trotter
// step: the on-point/corner extremes at half dtau must agree.
TEST(RegularizedCoulomb, EggBoxOfTheRelaxedGroundStateIsBelowBudget) {
    std::vector<double> e;
    for (const ses::Vec3d& c : kEggBoxOffsets) {
        e.push_back(relaxed_ground_energy(kEggBoxGrid, c, 0.05, 400));  // tau 20
        std::printf("  relaxed 1s c = (%.3f, %.3f, %.3f) h  E0 = %.6f Ha\n",
                    c.x / kEggBoxH, c.y / kEggBoxH, c.z / kEggBoxH, e.back());
    }
    const double de = spread(e);
    std::printf("  relaxed 1s egg-box spread = %.3e Ha\n", de);
    EXPECT_LT(de, 5.0e-4);
    // Absolute accuracy: the band-limited 1s lands within 2 mHa of -0.5 Ha
    // (cube average: -0.4913, point sampling: -0.4923).
    EXPECT_NEAR(e.front(), -0.5, 2.0e-3);
    const ses::Vec3d corner{0.5 * kEggBoxH, 0.5 * kEggBoxH, 0.5 * kEggBoxH};
    const double coarse = relaxed_ground_energy(kEggBoxGrid, corner, 0.05, 400) -
                          e.front();
    const double fine = relaxed_ground_energy(kEggBoxGrid, corner, 0.025, 800) -
                        relaxed_ground_energy(kEggBoxGrid, {}, 0.025, 800);
    std::printf("  on-point -> corner at dtau 0.05: %.3e, dtau 0.025: %.3e Ha\n",
                coarse, fine);
    EXPECT_NEAR(fine, coarse, 0.25 * std::abs(coarse));
}

// The residual must fall at least as h^2.6 (measured ~10x: 0.27 -> 0.026 mHa).
// Extremes only (on-point vs corner); 64^3 at h/2 is the costly half. The
// h/2 ground state is within 1 mHa of exact (non-variational: -0.5007).
TEST(RegularizedCoulomb, EggBoxOfTheRelaxedGroundStateConvergesWithSpacing) {
    const ses::Grid3D fine{ses::Grid1D{-5.0, 5.0, 64}, ses::Grid1D{-5.0, 5.0, 64},
                           ses::Grid1D{-5.0, 5.0, 64}};
    const double hf = 0.5 * kEggBoxH;
    // V dtau / 2 < 0.3 at the deeper -C/hf cell.
    const double coarse = spread({relaxed_ground_energy(kEggBoxGrid, {}, 0.05, 400),
                                  relaxed_ground_energy(
                                      kEggBoxGrid,
                                      {0.5 * kEggBoxH, 0.5 * kEggBoxH, 0.5 * kEggBoxH},
                                      0.05, 400)});
    const double refined = spread({relaxed_ground_energy(fine, {}, 0.03, 600),
                                   relaxed_ground_energy(
                                       fine, {0.5 * hf, 0.5 * hf, 0.5 * hf},
                                       0.03, 600)});
    std::printf("  egg-box on-point vs corner: h %.3e Ha, h/2 %.3e Ha\n", coarse,
                refined);
    EXPECT_LT(refined, 5.0e-5);
    EXPECT_LT(refined, coarse / 6.0);
    EXPECT_NEAR(relaxed_ground_energy(fine, {}, 0.03, 600), -0.5, 1.0e-3);
}

}  // namespace
