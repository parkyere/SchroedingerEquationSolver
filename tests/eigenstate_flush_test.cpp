// RED: post-collapse eigenstate-error flush.
// Collapse targets = sampled bare -Z/r radial eigenstates, NOT grid-H
// eigenstates (spectral kinetic + regularized Coulomb) -> Ha-scale cusp junk,
// so Var(H) > 0. A short ITP burst damps that junk (e^{-dE tau}, dE ~ Ha) fast
// but bound-bound admixture (dE <= 1 Ha) barely: it flushes error without
// draining identity or opening a non-radiative channel -- no-deflation license.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

import ses.imaginary_time;
import ses.observables;
import ses.radial;
import ses.grid;
import ses.vec;
import ses.field;
import ses.fft;
import ses.spectral;
import ses.harmonics;
import ses.potential;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::RadialGrid;
using ses::Vec3d;

// Coarse grid (as in regularized_coulomb_test) makes the cusp junk large --
// worst-case for the flush; box holds 1s and n=2 tails.
const Grid3D kGrid{Grid1D{-12.0, 12.0, 32}, Grid1D{-12.0, 12.0, 32},
                   Grid1D{-12.0, 12.0, 32}};

// Excited targets: 6 steps -- bounded on purpose (longer drains them downward).
// 1s target: 24 steps -- deep, cannot overshoot the ITP fixed point.
constexpr double kBurstDtau = 0.05;
constexpr int kBurstSteps = 6;
constexpr int kBurstStepsGround = 24;

const std::vector<double>& reg_potential() {
    static const std::vector<double> v =
        ses::regularized_coulomb_potential(kGrid, 1.0, Vec3d{});
    return v;
}

// Mirrors the app's collapse-target construction: bare -1/r radial solve
// sampled onto the grid.
Field3D synth(int l, int node_k, int m) {
    const RadialGrid rg{20.0, 3999};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = -1.0 / rg.r(i);
    }
    const ses::RadialState st =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, l), node_k);
    return ses::synthesize_orbital(kGrid, rg, st.u, l, m);
}

// The operator the split-step propagator exponentiates, applied once.
Field3D apply_h(const Field3D& f, const std::vector<double>& v) {
    Field3D t = f;
    ses::fft(t);
    const Grid3D& g = f.grid();
    const std::vector<double> kx = ses::wavenumbers(g.x);
    const std::vector<double> ky = ses::wavenumbers(g.y);
    const std::vector<double> kz = ses::wavenumbers(g.z);
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double kxx = kx[static_cast<std::size_t>(i)];
                const double kyy = ky[static_cast<std::size_t>(j)];
                const double kzz = kz[static_cast<std::size_t>(k)];
                t(i, j, k) *= 0.5 * (kxx * kxx + kyy * kyy + kzz * kzz);
            }
        }
    }
    ses::ifft(t);
    for (std::size_t i = 0; i < t.data().size(); ++i) {
        t.data()[i] += v[i] * f.data()[i];
    }
    return t;
}

// Scale-free: ratios cancel n2, so f need not be normalized. <H^2> = <Hf|Hf>
// by Hermiticity.
double energy_variance(const Field3D& f, const std::vector<double>& v) {
    const Field3D hf = apply_h(f, v);
    const double n2 = ses::norm_sq(f);
    const double e = ses::inner_product(f, hf).real() / n2;
    const double e2 = ses::norm_sq(hf) / n2;
    return e2 - e * e;
}

double population(const Field3D& phi, const Field3D& psi) {
    return std::norm(ses::inner_product(phi, psi));
}

void burst(Field3D& psi) {
    const ses::ImaginaryTimePropagator3D relaxer{kGrid, reg_potential(),
                                                 kBurstDtau};
    relaxer.relax(psi, kBurstSteps);  // renormalizes every step
}

TEST(EigenstateFlush, OneSBurstFlushesJunkTenfoldAndKeepsIdentity) {
    Field3D psi = synth(0, 0, 0);
    const Field3D psi0 = psi;
    const double var0 = energy_variance(psi, reg_potential());
    const double e0 = ses::mean_energy(psi, reg_potential());
    // The sampled 1s is measurably NOT a grid eigenstate (an eigenstate reads Var(H) = 0).
    ASSERT_GT(var0, 1e-3);

    burst(psi);

    const double var1 = energy_variance(psi, reg_potential());
    const double e1 = ses::mean_energy(psi, reg_potential());
    EXPECT_LT(var1, 0.1 * var0);          // junk flushed
    EXPECT_GT(population(psi0, psi), 0.97);  // identity preserved
    EXPECT_LT(e1, e0);                    // ITP is monotone toward the grid 1s
}

TEST(EigenstateFlush, GroundDeepBurstConvergesToTheGridGroundState) {
    Field3D psi = synth(0, 0, 0);
    const Field3D psi0 = psi;
    const ses::ImaginaryTimePropagator3D relaxer{kGrid, reg_potential(),
                                                 kBurstDtau};
    // Converged reference: the same flow run to its fixed point.
    Field3D ref = psi;
    relaxer.relax(ref, 400);
    const double e_ref = ses::mean_energy(ref, reg_potential());

    relaxer.relax(psi, kBurstStepsGround);

    EXPECT_NEAR(ses::mean_energy(psi, reg_potential()), e_ref, 5e-3);
    EXPECT_GT(population(psi0, psi), 0.99);
}

TEST(EigenstateFlush, TwoPzBurstOpensNoParityForbiddenOneSChannel) {
    Field3D psi = synth(1, 0, 0);
    const Field3D psi0 = psi;
    const Field3D s1 = synth(0, 0, 0);

    burst(psi);

    // No non-radiative 2p -> 1s decay: 2p_z is z-odd, H z-even, so the 1s
    // admixture stays at rounding; a fake jump channel would surface here first.
    EXPECT_LT(population(s1, psi), 1e-12);
    EXPECT_GT(population(psi0, psi), 0.99);  // 2p has no cusp: barely moves
}

TEST(EigenstateFlush, TwoSBurstDoesNotAmplifyGroundAdmixture) {
    Field3D psi = synth(0, 1, 0);
    const Field3D s1 = synth(0, 0, 0);
    const double p1_before = population(s1, psi);
    const double var0 = energy_variance(psi, reg_potential());

    burst(psi);

    // 2s shares the 1s parity sector -- worst case for no-deflation: at tau = 0.3
    // the 1s amplitude gains <= e^{0.35 tau} ~ 1.11, admixture stays same order.
    EXPECT_LT(population(s1, psi), 2.0 * p1_before + 1e-8);
    EXPECT_LT(energy_variance(psi, reg_potential()), 0.1 * var0);
}

}  // namespace
