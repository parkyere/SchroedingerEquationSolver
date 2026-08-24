// RED: time-dependent dipole drive V(r, t) = amplitude * (axis . r) * cos(omega t),
// inserted as scalar half-kicks around the static Strang step; the kicks are
// diagonal, so amplitude 0 leaves the static tables bitwise-identical.

#include <complex>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>
#include <algorithm>
#include <cstddef>
import ses.drive;
import ses.propagator;
import ses.observables;
import ses.grid;
import ses.vec;
import ses.field;
import ses.wavepacket;
import ses.potential;

#define SES_TEST_UTIL_HARMONIC
#include "test_util.h"

namespace {

using ses::DipoleDrive;
using ses::Field3D;
using ses::Grid3D;
using ses::Vec3d;
using ses_test::cube;

double population(const Field3D& state, const Field3D& psi) {
    const std::complex<double> ip = ses::inner_product(state, psi);
    return std::norm(ip);
}

TEST(DipoleDrive, ZeroAmplitudeMatchesStaticBitwise) {
    const Grid3D g = cube(-8.0, 8.0, 16);
    const std::vector<double> v = ses::harmonic_potential(g, 1.0, Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    const Field3D psi0 = ses::gaussian_wavepacket(g, Vec3d{1.0, 0.0, 0.0},
                                                  Vec3d{1.0, 1.0, 1.0}, Vec3d{});

    Field3D driven = psi0;
    ses::driven_step(driven, prop, DipoleDrive{Vec3d{0.0, 0.0, 1.0}, 0.0, 0.5}, 0.0, 10);
    Field3D fixed = psi0;
    prop.step(fixed, 10);

    EXPECT_EQ(ses_test::max_abs_diff(driven, fixed), 0.0);
}

TEST(DipoleDrive, ConstantFieldObeysEhrenfest) {
    // omega 0 -> constant force; wide grid keeps the packet clear of the wrap.
    const Grid3D g = cube(-16.0, 16.0, 32);
    const std::vector<double> zero_v(static_cast<std::size_t>(g.size()), 0.0);
    const double dt = 0.04;
    const ses::SplitOperator3D prop{g, zero_v, dt};
    Field3D psi = ses::gaussian_wavepacket(g, Vec3d{}, Vec3d{1.5, 1.5, 1.5}, Vec3d{});

    const double e0 = 0.05;
    const int steps = 50;
    ses::driven_step(psi, prop, DipoleDrive{Vec3d{0.0, 0.0, 1.0}, e0, 0.0}, 0.0, steps);

    const double t = steps * dt;
    EXPECT_NEAR(ses::mean_momentum(psi).z, -e0 * t, 1e-8);
    EXPECT_NEAR(ses::mean_position(psi).z, -0.5 * e0 * t * t, 1e-6);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);
}

TEST(DipoleDrive, ResonantCoherentLadderAndSelectionRule) {
    // Resonant z-drive climbs the whole equal-gap ladder -> coherent state, not two-level Rabi.
    const double w0 = 1.0;
    const Grid3D g = cube(-8.0, 8.0, 32);
    const std::vector<double> v = ses::harmonic_potential(g, w0, Vec3d{});
    const double dt = 0.05;
    const ses::SplitOperator3D prop{g, v, dt};

    const Field3D ground = ses_test::harmonic_state(g, w0, -1);
    const Field3D excited_z = ses_test::harmonic_state(g, w0, 2);
    const Field3D excited_y = ses_test::harmonic_state(g, w0, 1);

    const double e0 = 0.2;
    const int steps = 283;  // lands near a sin(w0 t) maximum
    const double t = steps * dt;

    Field3D psi = ground;
    ses::driven_step(psi, prop, DipoleDrive{Vec3d{0.0, 0.0, 1.0}, e0, w0}, 0.0, steps);

    // Ehrenfest exact for quadratic+linear V -> this closed form is the exact oracle.
    const double expected_z = -(e0 / (2.0 * w0)) * t * std::sin(w0 * t);
    EXPECT_NEAR(ses::mean_position(psi).z, expected_z, 0.02);

    // coherent-state Poisson populations
    const double alpha_sq = std::pow(e0 * t / (2.0 * std::sqrt(2.0 * w0)), 2.0);
    const double p0_expected = std::exp(-alpha_sq);
    const double p1_expected = alpha_sq * std::exp(-alpha_sq);
    EXPECT_NEAR(population(ground, psi), p0_expected, 0.06);
    EXPECT_NEAR(population(excited_z, psi), p1_expected, 0.06);

    // Selection rule: z-drive cannot populate the y state (<1_y|z|0> = 0 by symmetry).
    EXPECT_LT(population(excited_y, psi), 1e-4);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-10);
}

}  // namespace
