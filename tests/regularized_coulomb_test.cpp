// Spectrum contract for regularized_coulomb_potential; potential_test covers
// pointwise VALUES, this covers the resulting eigenspectrum.


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
import ses.harmonics;
import ses.wavepacket;
import ses.potential;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::RadialGrid;
using ses::Vec3d;

// Power-of-two per axis: hand-rolled radix-2 FFT requires it. +-12 Bohr box
// holds 1s tightly plus the n=2 tails.
const Grid3D kGrid{Grid1D{-12.0, 12.0, 32}, Grid1D{-12.0, 12.0, 32},
                   Grid1D{-12.0, 12.0, 32}};

const std::vector<double>& reg_potential() {
    static const std::vector<double> v =
        ses::regularized_coulomb_potential(kGrid, 1.0, Vec3d{});
    return v;
}

// <H> of a synthesized (u_nl/r) Y_lm orbital, mirroring the app path. Radial
// solve uses bare -1/r, differing from reg_potential only in the nucleus cell.
double synth_energy(int level_l, int node_k, int m) {
    const RadialGrid rg{20.0, 3999};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = -1.0 / rg.r(i);
    }
    const ses::RadialState st =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, level_l),
                               node_k);
    const Field3D psi = ses::synthesize_orbital(kGrid, rg, st.u, level_l, m);
    return ses::mean_energy(psi, reg_potential());
}

TEST(RegularizedCoulomb3D, GroundStateNearBareHydrogenAndDeeperThanSoft) {
    Field3D psi =
        ses::gaussian_wavepacket(kGrid, Vec3d{}, Vec3d{1.0, 1.0, 1.0}, Vec3d{});
    ses::ImaginaryTimePropagator3D relaxer{kGrid, reg_potential(), 0.02};
    relaxer.relax(psi, 900);  // tau = 18: converged to 1s
    const double e0 = ses::mean_energy(psi, reg_potential());
    EXPECT_LT(e0, 0.0);
    // -0.5 Ha = bare-hydrogen anchor; grid cusp relaxes to -0.473 (+0.027).
    // 0.08 band catches a wrong kCoulombCellAverage (halved -0.400, doubled -1.12).
    EXPECT_NEAR(e0, -0.5, 0.08);
    // Below the soft-Coulomb well (~-0.27 Ha): distinct regularizations, don't confuse.
    EXPECT_LT(e0, -0.35);
}

TEST(RegularizedCoulomb3D, TwoPTripletIsDegenerate) {
    // 2p_{x,y,z} degenerate under cubic-grid axis symmetry; nets axis-asymmetric bugs.
    const double e_pz = synth_energy(1, 0, 0);
    const double e_px = synth_energy(1, 0, 1);
    const double e_py = synth_energy(1, 0, -1);
    EXPECT_NEAR(e_px, e_pz, 1e-5);
    EXPECT_NEAR(e_py, e_pz, 1e-5);
}

TEST(RegularizedCoulomb3D, ShellOrderingAndN2Near) {
    const double e_1s = synth_energy(0, 0, 0);
    const double e_2s = synth_energy(0, 1, 0);
    const double e_2p = synth_energy(1, 0, 0);
    EXPECT_LT(e_1s, e_2s - 0.2);
    EXPECT_LT(e_1s, e_2p - 0.2);
    // n=2 shell near -1/8 = -0.125 Ha. 2s carries the coarse-grid cusp lift
    // that 2p (zero at nucleus) does not, so SO(4) 2s=2p holds only
    // approximately -> looser 2s band.
    EXPECT_NEAR(e_2p, -0.125, 0.03);
    EXPECT_NEAR(e_2s, -0.125, 0.08);
}

}  // namespace
