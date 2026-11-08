#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <random>
#include <vector>

import ses.spinlattice;

namespace {

ses::SpinLattice random_lattice(int n, unsigned seed) {
    ses::SpinLattice l;
    l.nx = n;
    l.ny = n;
    l.s.resize(static_cast<std::size_t>(n * n));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> ph(0.0, 2.0 * std::numbers::pi);
    for (auto& sp : l.s) {
        const double z = u(rng);
        const double t = ph(rng);
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        sp = ses::spinor_from_bloch(r * std::cos(t), r * std::sin(t), z);
    }
    return l;
}

TEST(SpinLattice, SpinorFromBlochRoundTrips) {
    const double r = 1.0 / std::sqrt(3.0);
    const ses::Spinor s = ses::spinor_from_bloch(r, -r, r);
    const ses::Vec3d p = ses::bloch_vector(s);
    EXPECT_NEAR(p.x, r, 1e-12);
    EXPECT_NEAR(p.y, -r, 1e-12);
    EXPECT_NEAR(p.z, r, 1e-12);
}

TEST(SpinLattice, DampedFerroOrdersDampedNeelStaggers) {
    ses::SpinLattice l = random_lattice(5, 11);
    for (int k = 0; k < 6000; ++k) {
        ses::spinlattice_step(l, 0.0, 0.0, 0.0, 0.5, 0.1, 0.05);
    }
    const ses::Vec3d m = ses::lattice_magnetization(l);
    EXPECT_GT(std::sqrt(m.x * m.x + m.y * m.y + m.z * m.z), 0.95);

    ses::SpinLattice a = random_lattice(5, 12);
    for (int k = 0; k < 12000; ++k) {
        ses::spinlattice_step(a, 0.0, 0.0, 0.0, -0.5, 0.1, 0.05);
    }
    const ses::Vec3d ma = ses::lattice_magnetization(a);
    EXPECT_LT(std::sqrt(ma.x * ma.x + ma.y * ma.y + ma.z * ma.z), 0.3);
    EXPECT_GT(ses::lattice_staggered(a), 0.95);
}

// Undamped: every site rotates about ITS OWN effective field while all of its
// neighbours (the other sublattice) are frozen, so n_i . B_eff,i -- and with
// it the mean-field energy -- is an EXACT invariant of each sweep. Any
// first-order drift means the rotation axis is not B_eff (a projected field
// walks the spin along a great circle instead of its Larmor cone).
TEST(SpinLattice, UndampedStepConservesTheMeanFieldEnergy) {
    ses::SpinLattice l = random_lattice(5, 13);
    const double e0 = ses::lattice_energy(l, 0.1, -0.2, 0.3, 0.4);
    for (int k = 0; k < 2000; ++k) {
        ses::spinlattice_step(l, 0.1, -0.2, 0.3, 0.4, 0.0, 0.01);
    }
    EXPECT_NEAR(ses::lattice_energy(l, 0.1, -0.2, 0.3, 0.4), e0, 1e-9);
}

TEST(SpinLattice, AlignedLatticePrecessesRigidly) {
    ses::SpinLattice l;
    l.nx = 5;
    l.ny = 5;
    l.s.assign(25, ses::spinor_from_bloch(1.0, 0.0, 0.0));
    const double b = 0.5;
    const double dt = 0.01;
    const int n = 400;
    for (int k = 0; k < n; ++k) {
        ses::spinlattice_step(l, 0.0, 0.0, b, 0.7, 0.0, dt);
    }
    // Parallel exchange = no torque -> no fan-out; staggered residual mz ~ 0.034.
    const ses::Vec3d m = ses::lattice_magnetization(l);
    EXPECT_GT(std::hypot(m.x, m.y), 0.995);  // no fan-out
    EXPECT_LT(std::abs(m.z), 0.06);
    EXPECT_NEAR(std::atan2(m.y, m.x), b * n * dt, 2e-3);  // +B CCW
}

}  // namespace
