// RED: pins the split-operator potential path e^{-i V dt/2}.
// Coherent state (ground state displaced by x_d), V = 1/2 omega^2 x^2, atomic units m=1:
//   <x>(t) = x_d cos(omega t),  <p>(t) = -x_d omega sin(omega t)
//   sigma_x = 1/sqrt(2 omega) const,  <H> = omega/2 + omega^2 x_d^2/2 const


#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>
import ses.propagator;
import ses.observables;
import ses.grid;
import ses.field;
import ses.wavepacket;
import ses.potential;

namespace {

using ses::Field1D;
using ses::Grid1D;
using ses::SplitOperator1D;

const Grid1D kGrid{-16.0, 16.0, 512};

constexpr double kOmega = 1.0;
constexpr double kXd = 2.0;
const double kSigmaGs = 1.0 / std::sqrt(2.0 * kOmega);
const double kDt = 2.0 * std::numbers::pi / 1000.0;

Field1D coherent_state() {
    return ses::gaussian_wavepacket(kGrid, kXd, kSigmaGs, 0.0);
}

SplitOperator1D harmonic_propagator(double dt) {
    return SplitOperator1D{kGrid, ses::harmonic_potential(kGrid, kOmega, 0.0), dt};
}

TEST(HarmonicDynamics, InitialEnergyIsAnalytic) {
    // omega/2 + omega^2 x_d^2/2 = 0.5 + 2 = 2.5
    const Field1D psi = coherent_state();
    const std::vector<double> v = ses::harmonic_potential(kGrid, kOmega, 0.0);
    EXPECT_NEAR(ses::mean_energy(psi, v), 2.5, 1e-8);
}

TEST(HarmonicDynamics, CoherentStateReachesMirrorPointAtHalfPeriod) {
    Field1D psi = coherent_state();
    harmonic_propagator(kDt).step(psi, 500);  // t = T/2 = pi
    EXPECT_NEAR(ses::mean_position(psi), -kXd, 1e-5);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);
}

TEST(HarmonicDynamics, CoherentStateDoesNotSpread) {
    Field1D psi = coherent_state();
    harmonic_propagator(kDt).step(psi, 500);  // t = pi
    EXPECT_NEAR(ses::sigma_x(psi), kSigmaGs, 1e-5);
}

TEST(HarmonicDynamics, MomentumPeaksAtQuarterPeriod) {
    Field1D psi = coherent_state();
    harmonic_propagator(kDt).step(psi, 250);  // t = T/4
    EXPECT_NEAR(ses::mean_momentum(psi), -kXd * kOmega, 1e-4);
    EXPECT_NEAR(ses::mean_position(psi), 0.0, 1e-4);
}

TEST(HarmonicDynamics, EnergyIsConservedOverFullPeriod) {
    Field1D psi = coherent_state();
    const std::vector<double> v = ses::harmonic_potential(kGrid, kOmega, 0.0);
    const double e0 = ses::mean_energy(psi, v);
    harmonic_propagator(kDt).step(psi, 1000);  // t = T
    EXPECT_NEAR(ses::mean_energy(psi, v), e0, 1e-4);
}

TEST(HarmonicDynamics, ConvergesAtSecondOrderInDt) {
    // Strang O(dt^2): halving dt drops <x> error ~4x
    const double t_final = 1.0;
    const double exact = kXd * std::cos(kOmega * t_final);

    auto error_with_dt = [&](double dt, int steps) {
        Field1D psi = coherent_state();
        harmonic_propagator(dt).step(psi, steps);
        return std::abs(ses::mean_position(psi) - exact);
    };

    const double e_coarse = error_with_dt(0.1, 10);
    const double e_fine = error_with_dt(0.05, 20);
    ASSERT_GT(e_coarse, 1e-9);  // guard: error must be measurable, not noise
    const double ratio = e_coarse / e_fine;
    EXPECT_GT(ratio, 3.4);
    EXPECT_LT(ratio, 4.6);
}

}  // namespace
