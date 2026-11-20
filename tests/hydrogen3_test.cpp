// 3D softened-hydrogen ground state; analytic oracles, no literature fits.


#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <vector>
import ses.imaginary_time;
import ses.observables;
import ses.grid;
import ses.vec;
import ses.field;
import ses.wavepacket;
import ses.potential;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

Grid3D cube(double lo, double hi, int n) {
    return Grid3D{Grid1D{lo, hi, n}, Grid1D{lo, hi, n}, Grid1D{lo, hi, n}};
}

const Grid3D kGrid = cube(-12.0, 12.0, 32);

struct Relaxed {
    Field3D psi;
    double energy;
    double seed_energy;  // <H> of the Gaussian the relaxation started from
};

Relaxed relax_hydrogen(double a) {
    const std::vector<double> v = ses::soft_coulomb_potential(kGrid, 1.0, a, Vec3d{});
    Field3D psi = ses::gaussian_wavepacket(kGrid, Vec3d{}, Vec3d{1.3, 1.3, 1.3}, Vec3d{});
    const double e_seed = ses::mean_energy(psi, v);
    ses::ImaginaryTimePropagator3D relaxer{kGrid, v, 0.02};
    relaxer.relax(psi, 750);  // tau = 15
    const double e = ses::mean_energy(psi, v);
    return Relaxed{psi, e, e_seed};
}

const Relaxed& hydrogen_a1() {
    static const Relaxed r = relax_hydrogen(1.0);
    return r;
}

TEST(Hydrogen3, GroundStateEnergyWithinVariationalWindow) {
    const double e0 = hydrogen_a1().energy;
    EXPECT_GT(e0, -0.5);  // above the exact bare -1/r ground state
    EXPECT_LT(e0, -0.1);
}

// The seed alone already sits inside that window and is spherical and
// centred, so the tests around this one would pass with relax() deleted.
// What the relaxation must do: go below its seed (variational; measured
// -0.25489 -> -0.27491) and stay put under more imaginary time (tau = 5
// more moves E by 3e-5: converged to the 1e-4 the gate asks).
TEST(Hydrogen3, RelaxationLowersTheSeedEnergyAndConverges) {
    const Relaxed& r = hydrogen_a1();
    std::printf("hydrogen3 a=1: seed E %.5f -> relaxed E %.5f\n", r.seed_energy,
                r.energy);
    EXPECT_LT(r.energy, r.seed_energy - 0.015);
    const std::vector<double> v =
        ses::soft_coulomb_potential(kGrid, 1.0, 1.0, Vec3d{});
    Field3D more = r.psi;
    ses::ImaginaryTimePropagator3D relaxer{kGrid, v, 0.02};
    relaxer.relax(more, 250);  // tau = 5 more
    EXPECT_NEAR(ses::mean_energy(more, v), r.energy, 1e-4);
}

TEST(Hydrogen3, EnergyIncreasesWithSoftening) {
    const double e_soft = relax_hydrogen(1.4).energy;
    const double e_hard = hydrogen_a1().energy;
    // variational monotonicity: shallower V -> higher E0; same-grid bias cancels
    EXPECT_LT(e_hard, e_soft);
    EXPECT_GT(e_soft - e_hard, 0.01);
}

TEST(Hydrogen3, CloudIsSphericalAboutNucleus) {
    const Field3D& psi = hydrogen_a1().psi;
    // Grid [-12,12) asymmetric about 0 biases every axis' mean identically;
    // axis EQUALITY is the sharp oracle (1e-9), absolute center looser (1e-4).
    const Vec3d r = ses::mean_position(psi);
    EXPECT_NEAR(r.x, 0.0, 1e-4);
    EXPECT_NEAR(r.x, r.y, 1e-9);
    EXPECT_NEAR(r.y, r.z, 1e-9);
    const Vec3d s = ses::sigma_position(psi);
    EXPECT_NEAR(s.x, s.y, 1e-9);
    EXPECT_NEAR(s.y, s.z, 1e-9);
    EXPECT_GT(s.x, 0.5);  // extended cloud, not a collapsed spike
}

}  // namespace
