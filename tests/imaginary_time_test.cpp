// RED: imaginary-time relaxation to the ground state.


#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
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

// relax() IS the Strang step of the public blocks -- half weight, transform,
// kinetic weight, inverse, half weight, renormalize -- bit for bit. Pins the
// result ahead of a change in how the propagator holds its FFT tables.
TEST(ImaginaryTime, RelaxIsTheStrangStepOfThePublicBlocksBitwise) {
    const std::vector<double> v = ses::harmonic_potential(kGrid, 1.0, 0.0);
    const double dtau = 0.005;
    const int steps = 7;
    Field1D psi = ses::gaussian_wavepacket(kGrid, 1.0, 2.0, 0.3);
    Field1D ref = psi;
    const ImaginaryTimePropagator1D relaxer{kGrid, v, dtau};
    relaxer.relax(psi, steps);

    const auto weight = [](double th) { return std::exp(th); };
    const std::vector<double> half =
        ses::build_half_potential_table(v, dtau, weight);
    const std::vector<double> kin =
        ses::build_kinetic_table(kGrid, 1.0, dtau, weight);
    std::vector<std::complex<double>>& a = ref.data();
    for (int s = 0; s < steps; ++s) {
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = half[i] * a[i];
        }
        ses::fft(a);
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = kin[i] * a[i];
        }
        ses::ifft(a);
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = half[i] * a[i];
        }
        ses::normalize(ref);
    }
    EXPECT_EQ(psi.data(), ref.data());
}

}  // namespace
