// CPU logic (clip ray-interval, slice quad, slice colour) is the source of truth
// that volume.frag / slice.vert / slice.frag mirror textually; no per-fragment GPU
// oracle exists, so transcription is verified by the dump-frame composite.

#include <algorithm>
#include <complex>

#include <gtest/gtest.h>

#include <cmath>
import ses.grid;
import ses.vec;
import ses.colormap;
import ses.cross_section;


namespace {

using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

TEST(ClipRayInterval, ClampsToTheVisibleHalfCrossingForward) {
    // sign*(p-offset)<=0, sign=+1 offset=0 => p<=0 => t<=5; t_stop 10->5
    const ses::ClipInterval c = ses::clip_ray_interval(0.0, 10.0, -5.0, 1.0,
                                                       1.0, 0.0);
    EXPECT_TRUE(c.visible);
    EXPECT_NEAR(c.tn, 0.0, 1e-12);
    EXPECT_NEAR(c.t_stop, 5.0, 1e-12);
}

TEST(ClipRayInterval, SignFlipKeepsTheOtherHalf) {
    const ses::ClipInterval c = ses::clip_ray_interval(0.0, 10.0, -5.0, 1.0,
                                                       -1.0, 0.0);
    EXPECT_TRUE(c.visible);
    EXPECT_NEAR(c.tn, 5.0, 1e-12);
    EXPECT_NEAR(c.t_stop, 10.0, 1e-12);
}

TEST(ClipRayInterval, OffsetShiftsThePlane) {
    const ses::ClipInterval c = ses::clip_ray_interval(0.0, 10.0, -5.0, 1.0,
                                                       1.0, 2.0);
    EXPECT_NEAR(c.t_stop, 7.0, 1e-12);  // p <= 2 => t <= 7
}

TEST(ClipRayInterval, ParallelRayOnVisibleSideIsUnchanged) {
    const ses::ClipInterval c = ses::clip_ray_interval(1.0, 9.0, -3.0, 0.0,
                                                       1.0, 0.0);
    EXPECT_TRUE(c.visible);
    EXPECT_NEAR(c.tn, 1.0, 1e-12);
    EXPECT_NEAR(c.t_stop, 9.0, 1e-12);
}

TEST(ClipRayInterval, ParallelRayOnCutSideIsInvisible) {
    const ses::ClipInterval c = ses::clip_ray_interval(1.0, 9.0, 3.0, 0.0,
                                                       1.0, 0.0);
    EXPECT_FALSE(c.visible);
}

// The non-parallel branch can hide a ray too: an eye at p = 5 on the cut side
// moving +p reaches the plane only at t = -5, before tn = 0.
TEST(ClipRayInterval, RayMovingAwayFromTheVisibleHalfIsInvisible) {
    const ses::ClipInterval c = ses::clip_ray_interval(0.0, 10.0, 5.0, 1.0,
                                                       1.0, 0.0);
    EXPECT_FALSE(c.visible);
}

TEST(SliceQuadCorner, AllSixCornersLieOnThePlaneAndSpanTheBox) {
    const Grid1D ax{-8.0, 8.0, 16};
    const Grid3D g{ax, ax, ax};
    const int axis = 2;
    const double offset = 3.0;
    double umin = 1e9, umax = -1e9, wmin = 1e9, wmax = -1e9;
    for (int k = 0; k < 6; ++k) {
        const Vec3d p = ses::slice_quad_corner(axis, offset, g, k);
        EXPECT_NEAR(p.z, offset, 1e-12);
        umin = std::min(umin, p.x);
        umax = std::max(umax, p.x);
        wmin = std::min(wmin, p.y);
        wmax = std::max(wmax, p.y);
    }
    EXPECT_NEAR(umin, -8.0, 1e-12);
    EXPECT_NEAR(umax, 8.0, 1e-12);
    EXPECT_NEAR(wmin, -8.0, 1e-12);
    EXPECT_NEAR(wmax, 8.0, 1e-12);
}

TEST(SliceShade, DensityModeIsMagnitudeColorAndBaseAlpha) {
    // alpha = 0.45 + 0.5*0.5 = 0.70
    const std::complex<double> psi{std::sqrt(0.5), 0.0};
    const ses::SliceShade s = ses::slice_shade(ses::SliceMap::Density, psi, 1.0);
    const ses::Rgb m = ses::magnitude_color(0.5);
    EXPECT_NEAR(s.col.r, m.r, 1e-12);
    EXPECT_NEAR(s.col.g, m.g, 1e-12);
    EXPECT_NEAR(s.col.b, m.b, 1e-12);
    EXPECT_NEAR(s.alpha, 0.70, 1e-12);
}

TEST(SliceShade, ReModeIsDivergingBySign) {
    const ses::SliceShade pos = ses::slice_shade(
        ses::SliceMap::Real, std::complex<double>{0.8, 0.0}, 1.0);
    const ses::SliceShade neg = ses::slice_shade(
        ses::SliceMap::Real, std::complex<double>{-0.8, 0.0}, 1.0);
    EXPECT_GT(pos.col.r, pos.col.b);
    EXPECT_GT(neg.col.b, neg.col.r);
    EXPECT_NEAR(pos.alpha, 0.85, 1e-12);  // 0.45 + 0.5*0.8
    // zero amplitude -> dark diverging midpoint
    const ses::SliceShade zero = ses::slice_shade(
        ses::SliceMap::Real, std::complex<double>{0.0, 0.0}, 1.0);
    EXPECT_NEAR(zero.col.r, 0.03, 1e-12);
    EXPECT_NEAR(zero.col.b, 0.03, 1e-12);
}

// inv_peak = 1/peak density normalizes the three maps: the density scales
// LINEARLY with it, the real amplitude with its SQUARE ROOT. At inv_peak = 1
// (every other test here) the two are indistinguishable; slice.frag mirrors
// both lines and has no GPU oracle. Peak density 1/4 -> inv_peak 4.
TEST(SliceShade, PeakScalesDensityLinearlyAndAmplitudeByItsRoot) {
    const double inv_peak = 4.0;
    // |psi|^2 = 1/16 -> dens = 1/4 -> alpha = 0.45 + 0.5/4
    const ses::SliceShade dens = ses::slice_shade(
        ses::SliceMap::Density, std::complex<double>{0.25, 0.0}, inv_peak);
    const ses::Rgb m = ses::magnitude_color(0.25);
    EXPECT_NEAR(dens.col.r, m.r, 1e-12);
    EXPECT_NEAR(dens.col.g, m.g, 1e-12);
    EXPECT_NEAR(dens.col.b, m.b, 1e-12);
    EXPECT_NEAR(dens.alpha, 0.575, 1e-12);
    // Re psi = 1/4 -> r = 1/4 * sqrt(4) = 1/2 -> alpha = 0.45 + 0.5/2
    const ses::SliceShade re = ses::slice_shade(
        ses::SliceMap::Real, std::complex<double>{0.25, 0.0}, inv_peak);
    EXPECT_NEAR(re.alpha, 0.70, 1e-12);
    EXPECT_NEAR(re.col.r, 0.03 + (1.0 - 0.03) * 0.5, 1e-12);
    // phase map: bright = sqrt(dens) = 1/2 -> alpha 0.70, tint 0.625
    const ses::SliceShade ph = ses::slice_shade(
        ses::SliceMap::Phase, std::complex<double>{0.0, 0.25}, inv_peak);
    const ses::Rgb wheel = ses::phase_color(std::atan2(0.25, 0.0));
    EXPECT_NEAR(ph.alpha, 0.70, 1e-12);
    EXPECT_NEAR(ph.col.r, wheel.r * 0.625, 1e-12);
    EXPECT_NEAR(ph.col.g, wheel.g * 0.625, 1e-12);
    EXPECT_NEAR(ph.col.b, wheel.b * 0.625, 1e-12);
}

TEST(SliceShade, PhaseModeTintsThePhaseWheelByMagnitude) {
    const double mag = std::sqrt(0.25);  // dens = 0.25, bright = sqrt(dens) = 0.5
    const std::complex<double> psi{0.0, mag};
    const ses::SliceShade s = ses::slice_shade(ses::SliceMap::Phase, psi, 1.0);
    const ses::Rgb wheel = ses::phase_color(std::atan2(mag, 0.0));
    const double bright = 0.5;
    const double tint = 0.25 + 0.75 * bright;
    EXPECT_NEAR(s.col.r, wheel.r * tint, 1e-12);
    EXPECT_NEAR(s.col.g, wheel.g * tint, 1e-12);
    EXPECT_NEAR(s.col.b, wheel.b * tint, 1e-12);
    EXPECT_NEAR(s.alpha, 0.45 + 0.5 * bright, 1e-12);
}

}  // namespace
