// RED: imaginary-time relaxation to the ground state.


#include <gtest/gtest.h>

#include <cmath>
#include <vector>
import ses.imaginary_time;
import ses.observables;
import ses.grid;
import ses.field;
import ses.wavepacket;
import ses.potential;

namespace {

using ses::Field1D;
using ses::Grid1D;
using ses::ImaginaryTimePropagator1D;

const Grid1D kGrid{-16.0, 16.0, 512};

TEST(ImaginaryTime, RelaxedStateIsNormalized) {
    const std::vector<double> v = ses::harmonic_potential(kGrid, 1.0, 0.0);
    Field1D psi = ses::gaussian_wavepacket(kGrid, 1.0, 2.0, 0.0);
    ImaginaryTimePropagator1D relaxer{kGrid, v, 0.005};
    relaxer.relax(psi, 100);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);
}

TEST(ImaginaryTime, FindsHarmonicGroundState) {
    const double omega = 1.0;
    const std::vector<double> v = ses::harmonic_potential(kGrid, omega, 0.0);
    // deliberately wrong start (x0=1, sigma=2); relaxer must forget both
    Field1D psi = ses::gaussian_wavepacket(kGrid, 1.0, 2.0, 0.0);
    ImaginaryTimePropagator1D relaxer{kGrid, v, 0.005};
    relaxer.relax(psi, 6000);  // tau = 30 >> 1/(E1-E0) = 1
    EXPECT_NEAR(ses::mean_energy(psi, v), 0.5 * omega, 1e-5);
    EXPECT_NEAR(ses::sigma_x(psi), 1.0 / std::sqrt(2.0 * omega), 1e-5);
    EXPECT_NEAR(ses::mean_position(psi), 0.0, 1e-8);
}

TEST(ImaginaryTime, FindsSoftCoulombBoundState) {
    // V = -1/sqrt(x^2+1)
    const std::vector<double> v = ses::soft_coulomb_potential(kGrid, 1.0, 1.0, 0.0);
    Field1D psi = ses::gaussian_wavepacket(kGrid, 0.5, 1.5, 0.0);
    ImaginaryTimePropagator1D relaxer{kGrid, v, 0.005};
    relaxer.relax(psi, 10000);
    const double e0 = ses::mean_energy(psi, v);
    EXPECT_LT(e0, -0.5);
    EXPECT_NEAR(e0, -0.6698, 1e-3);   // literature value
    EXPECT_NEAR(ses::mean_position(psi), 0.0, 1e-6);
}

}  // namespace
