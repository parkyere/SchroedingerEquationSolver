// RED: exact prolate-spheroidal H2+ eigensolver (ses.spheroidal).
// Oracle E_elec (electronic, excludes 1/R) at R=2; Scott arXiv:physics/0607081,
// Turbiner arXiv:1401.8009.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

import ses.spheroidal;
import ses.h2plus_atlas_loader;
import ses.field;
import ses.grid;
import ses.observables;
import ses.potential;
import ses.vec;
import ses.imaginary_time;
import ses.spectral;
import ses.fft;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

// Tolerance loose: coarse FD solve.
TEST(Spheroidal, KnownOrbitalEnergiesAtEquilibrium) {
    const double R = 2.0;
    const ses::H2plusOrbital sg = ses::h2plus_orbital(R, 0, 0, 0);
    EXPECT_NEAR(sg.energy, -1.1026342, 0.01) << "1sigma_g electronic";
    EXPECT_EQ(sg.parity, +1) << "1sigma_g is gerade";

    const ses::H2plusOrbital su = ses::h2plus_orbital(R, 0, 1, 0);
    EXPECT_NEAR(su.energy, -0.6675344, 0.01) << "2p sigma_u* electronic";
    EXPECT_EQ(su.parity, -1) << "2p sigma_u* is ungerade";

    const ses::H2plusOrbital sg2 = ses::h2plus_orbital(R, 0, 0, 1);
    EXPECT_NEAR(sg2.energy, -0.3608649, 0.01) << "2sigma_g electronic";
    EXPECT_EQ(sg2.parity, +1);

    const ses::H2plusOrbital pu = ses::h2plus_orbital(R, 1, 0, 0);
    EXPECT_NEAR(pu.energy, -0.4287723, 0.01) << "1pi_u electronic";
    EXPECT_EQ(pu.parity, -1) << "1pi_u is ungerade";

    EXPECT_LT(sg.energy, su.energy);
    EXPECT_LT(su.energy, pu.energy);
    EXPECT_LT(pu.energy, sg2.energy);
}

// Contract: a converged solve reports valid; the flag gates h2plus_atlas, so
// losing it would silently drop every state.
TEST(Spheroidal, SolvedOrbitalReportsValid) {
    EXPECT_TRUE(ses::h2plus_orbital(2.0, 0, 0, 0).valid);
    EXPECT_TRUE(ses::h2plus_orbital(2.0, 1, 0, 0).valid);
}

TEST(Spheroidal, UnbracketedRootReportsInvalid) {
    // n_xi = 6 at R = 0.5: g(p^2) never changes sign below the p^2 ceiling,
    // so p2 is a stale midpoint (E ~ -33 Ha, NEGATIVE: the old energy-only
    // atlas filter would have admitted this garbage). The flag must say so.
    EXPECT_FALSE(ses::h2plus_orbital(0.5, 0, 1, 6).valid);
}

// RED: synthesize_h2plus along an ARBITRARY molecular axis n with azimuth
// reference e1 (e2 = n x e1). The legacy call is axis = x-hat, e1 = y-hat
// (phi = atan2(z, y)); the rotor scene needs the orbital along its live
// axis. Oracle: for n = z-hat, e1 = y-hat the field is the x-axis field
// under (x, y, z) -> (z, y, -x): f_z(i, j, k) == f_x(k, j, n - i) on the
// symmetric grid (i >= 1: index 0 has no mirror on a periodic axis).
TEST(SynthesizeH2plus, GeneralAxisMatchesTheXAxisFieldUnderRotation) {
    const int n = 24;
    const ses::Grid1D axis{-4.0, 4.0, n};
    const ses::Grid3D g{axis, axis, axis};
    for (const int m : {0, 1}) {
        for (const int partner : {0, 1}) {
            if (m == 0 && partner == 1) {
                continue;  // m = 0 ignores the partner
            }
            const ses::H2plusOrbital o = ses::h2plus_orbital(2.0, m, 0, 0);
            const ses::Field3D fx = ses::synthesize_h2plus(g, o, partner);
            const ses::Field3D fz = ses::synthesize_h2plus(
                g, o, partner, ses::Vec3d{0.0, 0.0, 1.0}, ses::Vec3d{0.0, 1.0, 0.0});
            double max_err = 0.0;
            for (int k = 0; k < n; ++k) {
                for (int j = 0; j < n; ++j) {
                    for (int i = 1; i < n; ++i) {
                        max_err = std::max(
                            max_err,
                            std::abs(fz(i, j, k).real() - fx(k, j, n - i).real()));
                    }
                }
            }
            EXPECT_LT(max_err, 1e-12) << "m = " << m << " partner = " << partner;
        }
    }
}

TEST(SynthesizeH2plus, DefaultAxisIsBitwiseTheLegacyXCall) {
    const ses::Grid1D axis{-4.0, 4.0, 16};
    const ses::Grid3D g{axis, axis, axis};
    const ses::H2plusOrbital o = ses::h2plus_orbital(2.0, 1, 0, 0);
    const ses::Field3D legacy = ses::synthesize_h2plus(g, o, 1);
    const ses::Field3D general = ses::synthesize_h2plus(
        g, o, 1, ses::Vec3d{1.0, 0.0, 0.0}, ses::Vec3d{0.0, 1.0, 0.0});
    for (std::size_t c = 0; c < legacy.data().size(); ++c) {
        ASSERT_EQ(general.data()[c], legacy.data()[c]) << "cell " << c;
    }
}

TEST(Spheroidal, GroundEnergyVsInternuclearDistance) {
    // Oracle: Turbiner Table I.
    EXPECT_NEAR(ses::h2plus_orbital(1.0, 0, 0, 0).energy, -1.4517863, 0.02);
    EXPECT_NEAR(ses::h2plus_orbital(2.0, 0, 0, 0).energy, -1.1026342, 0.01);
    EXPECT_NEAR(ses::h2plus_orbital(4.0, 0, 0, 0).energy, -0.7960849, 0.01);

    const double et1 = ses::h2plus_orbital(1.0, 0, 0, 0).energy + 1.0 / 1.0;
    const double et2 = ses::h2plus_orbital(2.0, 0, 0, 0).energy + 1.0 / 2.0;
    const double et4 = ses::h2plus_orbital(4.0, 0, 0, 0).energy + 1.0 / 4.0;
    EXPECT_LT(et2, et1) << "the bond binds vs compressed";
    EXPECT_LT(et2, et4) << "the bond binds vs stretched";
}

// Energy checked only loosely (coarse-grid two-cusp resolution gap); shape checks pin the synthesis.
TEST(Spheroidal, SynthesizedGroundIsGeradeAndOnTheNuclei) {
    const double R = 2.0;
    const Grid1D ax{-16.0, 16.0, 128};
    const Grid3D g{ax, ax, ax};
    const ses::H2plusOrbital sg = ses::h2plus_orbital(R, 0, 0, 0);
    const Field3D psi = ses::synthesize_h2plus(g, sg, 0);

    auto nearest = [](const Grid1D& a, double x) {
        int best = 0;
        for (int i = 1; i < a.n; ++i) {
            if (std::abs(a.coord(i) - x) < std::abs(a.coord(best) - x)) {
                best = i;
            }
        }
        return best;
    };
    const int cy = nearest(g.y, 0.0);
    const int nx = nearest(g.x, R / 2);
    const int fx = nearest(g.x, 10.0);
    EXPECT_GT(std::norm(psi(nx, cy, cy)), 100.0 * std::norm(psi(fx, cy, cy)));
    EXPECT_NEAR(psi(nx, cy, cy).real(), psi(g.x.n - nx, cy, cy).real(),
                1e-6 * std::abs(psi(nx, cy, cy).real()) + 1e-9);

    const std::vector<double> v = ses::regularized_coulomb_potential(
        g, 1.0, {{-R / 2, 0.0, 0.0}, {R / 2, 0.0, 0.0}});
    const double e = ses::mean_energy(psi, v);
    EXPECT_LT(e, -0.5) << "clearly bound";
    EXPECT_GT(e, sg.energy - 0.1) << "not spuriously deeper than the exact";
}

TEST(Spheroidal, PiUOrbitalHasAnAxisNode) {
    const double R = 2.0;
    const Grid1D ax{-16.0, 16.0, 128};
    const Grid3D g{ax, ax, ax};
    const ses::H2plusOrbital pu = ses::h2plus_orbital(R, 1, 0, 0);
    // partner 0 = cos(phi): nodal plane y=0.
    const Field3D psi = ses::synthesize_h2plus(g, pu, 0);
    double node = 0.0;
    double bulk = 0.0;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double w = std::norm(psi(i, j, k));
                if (j == 64) {
                    node += w;
                } else {
                    bulk += w;
                }
            }
        }
    }
    EXPECT_LT(node, 0.02 * bulk) << "1pi_u (cos phi) has the y = 0 nodal plane";
}

TEST(Spheroidal, BakedAtlasMatchesTheLiveSolve) {
    // R=1.875 = exact baked grid point (2h snap at 256^3/+-30).
    const double R = 1.875;
    const std::vector<ses::H2plusOrbital> baked = ses::h2plus_atlas_baked(R);
    ASSERT_FALSE(baked.empty());
    const std::vector<ses::H2plusOrbital> live =
        ses::h2plus_atlas(R, static_cast<int>(baked.size()));
    ASSERT_EQ(baked.size(), live.size());
    for (std::size_t i = 0; i < baked.size(); ++i) {
        EXPECT_EQ(baked[i].m, live[i].m);
        EXPECT_EQ(baked[i].parity, live[i].parity);
        EXPECT_NEAR(baked[i].energy, live[i].energy, 1e-6)
            << "baked orbital " << i << " energy";
    }
    EXPECT_LT(baked[0].energy, -1.0);
}

TEST(Spheroidal, BakedGroundSynthesizesLikeTheLiveSolve) {
    const double R = 2.0;
    const Grid1D ax{-16.0, 16.0, 128};
    const Grid3D g{ax, ax, ax};
    const ses::H2plusOrbital b0 = ses::h2plus_atlas_baked(R).front();
    const Field3D psi = ses::synthesize_h2plus(g, b0, 0);
    const std::vector<double> v = ses::regularized_coulomb_potential(
        g, 1.0, {{-R / 2, 0.0, 0.0}, {R / 2, 0.0, 0.0}});
    const double e = ses::mean_energy(psi, v);
    EXPECT_LT(e, -0.5) << "baked ground synthesizes a bound state";
}

// ---- atlas flush (CONTRACT for the H2+ scene's prepare(k)) ----
// A spheroidal orbital SAMPLED on h = 0.3125 is not a grid eigenstate: the
// cusp's high-k content puts 1sigma_g 145 mHa above the grid ground (Var 1.8
// Ha^2, 3.7% continuum that disperses over the box in ~40 au). The scene's
// flush -- ITP kH2plusAtlasFlushSteps x kH2plusAtlasFlushDtau, deflated
// against the lower SYNTHESIZED members -- must land each member on its grid
// state: |E - E_grid| < 1 mHa, overlap > 0.999, Var < 1e-2.

double energy_variance3(const Field3D& psi, const std::vector<double>& v) {
    const Grid3D& g = psi.grid();
    Field3D hp = psi;
    ses::fft(hp);
    const std::vector<double> kx = ses::wavenumbers(g.x);
    const std::vector<double> ky = ses::wavenumbers(g.y);
    const std::vector<double> kz = ses::wavenumbers(g.z);
    ses::for_each_cell(g, [&](int i, int j, int k) {
        hp(i, j, k) *= 0.5 * (kx[i] * kx[i] + ky[j] * ky[j] + kz[k] * kz[k]);
    });
    ses::ifft(hp);
    double hh = 0.0;
    double nn = 0.0;
    for (std::size_t i = 0; i < psi.data().size(); ++i) {
        hh += std::norm(hp.data()[i] + v[i] * psi.data()[i]);
        nn += std::norm(psi.data()[i]);
    }
    const double e = ses::mean_energy(psi, v);
    return hh / nn - e * e;
}

struct AtlasFlushRig {
    Grid3D g{Grid1D{-10.0, 10.0, 64}, Grid1D{-10.0, 10.0, 64},
             Grid1D{-10.0, 10.0, 64}};  // h = 0.3125 = the 256^3/+-40 scene
    double R = 1.875;
    std::vector<double> v = ses::regularized_coulomb_potential(
        g, 1.0, std::vector<Vec3d>{{0.0, 0.0, 0.5 * R}, {0.0, 0.0, -0.5 * R}});
    std::vector<ses::H2plusOrbital> atlas = ses::h2plus_atlas_baked(R);
    Field3D synth(int k) const {
        Field3D f = ses::synthesize_h2plus(g, atlas[static_cast<std::size_t>(k)], 0,
                                           Vec3d{0.0, 0.0, 1.0}, Vec3d{1.0, 0.0, 0.0});
        ses::normalize(f);
        return f;
    }
};

TEST(AtlasFlush, SynthesizedSigmaGLandsOnTheGridGround) {
    const AtlasFlushRig rig;
    const ses::ImaginaryTimePropagator3D itp{rig.g, rig.v, ses::kH2plusAtlasFlushDtau};
    const Field3D sg = rig.synth(0);
    Field3D ground = sg;
    itp.relax(ground, 400);  // converged reference (tau = 20)
    const double e_grid = ses::mean_energy(ground, rig.v);
    // Before: not a grid eigenstate (the defect the flush exists for).
    EXPECT_GT(ses::mean_energy(sg, rig.v) - e_grid, 0.1);
    EXPECT_LT(std::norm(ses::inner_product(ground, sg)), 0.97);
    Field3D psi = sg;
    itp.relax(psi, ses::kH2plusAtlasFlushSteps);
    const double e = ses::mean_energy(psi, rig.v);
    std::printf("  sigma_g: E %.5f -> %.5f (grid %.5f), Var %.2e, overlap %.5f\n",
                ses::mean_energy(sg, rig.v), e, e_grid, energy_variance3(psi, rig.v),
                std::norm(ses::inner_product(ground, psi)));
    EXPECT_NEAR(e, e_grid, 1e-3);
    EXPECT_GT(std::norm(ses::inner_product(ground, psi)), 0.999);
    EXPECT_LT(energy_variance3(psi, rig.v), 1e-2);
}

TEST(AtlasFlush, ExcitedMemberDeflatedAgainstTheSynthesizedLowerOnes) {
    const AtlasFlushRig rig;
    const ses::ImaginaryTimePropagator3D itp{rig.g, rig.v, ses::kH2plusAtlasFlushDtau};
    const Field3D sg = rig.synth(0);
    const Field3D su = rig.synth(1);
    Field3D ground = sg;
    itp.relax(ground, 400);
    Field3D excited = su;
    itp.relax_deflated(excited, {&ground}, 400);  // converged 1sigma_u* reference
    const double e_grid = ses::mean_energy(excited, rig.v);
    EXPECT_GT(ses::mean_energy(su, rig.v) - e_grid, 0.02);
    // The scene deflates against the synthesized (not flushed) lower members.
    Field3D psi = su;
    itp.relax_deflated(psi, {&sg}, ses::kH2plusAtlasFlushSteps);
    const double e = ses::mean_energy(psi, rig.v);
    std::printf("  sigma_u*: E %.5f -> %.5f (grid %.5f), overlap %.5f\n",
                ses::mean_energy(su, rig.v), e, e_grid,
                std::norm(ses::inner_product(excited, psi)));
    EXPECT_NEAR(e, e_grid, 1e-3);
    EXPECT_GT(std::norm(ses::inner_product(excited, psi)), 0.999);
    EXPECT_LT(energy_variance3(psi, rig.v), 1e-2);
}


// Flushed reference chain: member k relaxed 400 steps deflated against the
// converged lower members (the grid's own bound states, in atlas order).
std::vector<Field3D> flushed_reference_chain(const AtlasFlushRig& rig, int top) {
    const ses::ImaginaryTimePropagator3D itp{rig.g, rig.v, ses::kH2plusAtlasFlushDtau};
    std::vector<Field3D> ref;
    for (int k = 0; k < top; ++k) {
        std::vector<const Field3D*> lower;
        for (const Field3D& r : ref) {
            lower.push_back(&r);
        }
        Field3D f = rig.synth(k);
        itp.relax_deflated(f, lower, 400);
        ref.push_back(std::move(f));
    }
    return ref;
}

// The scene exposes up to 24 members. For k >= 2 the flush cannot reach the
// converged grid state at tau = 2: a sampled member carries ~0.5% of the
// SAME-symmetry higher bound states (2p pi_u holds 3p pi_u, dE = 0.23 Ha ->
// e^{-0.47} per flush; pre-flushing the projectors changes nothing, the
// lower sigma members are orthogonal by symmetry). What the flush promises
// for every member: the continuum is gone (Var < 1e-2), E within 3 mHa of
// the converged grid state, overlap > 0.99 (k <= 1: 1 mHa, 0.999).
TEST(AtlasFlush, EveryExposedMemberLandsOnItsGridState) {
    const AtlasFlushRig rig;
    const int top = 5;  // 1s sg, 1s su*, 2p pu, 2s sg, 2p su
    const std::vector<Field3D> ref = flushed_reference_chain(rig, top);
    std::vector<Field3D> synth;
    for (int k = 0; k < top; ++k) {
        synth.push_back(rig.synth(k));
    }
    const ses::ImaginaryTimePropagator3D itp{rig.g, rig.v, ses::kH2plusAtlasFlushDtau};
    for (int k = 0; k < top; ++k) {
        // Projectors = the raw synthesized lower members (what the scene does).
        std::vector<const Field3D*> lower;
        for (int j = 0; j < k; ++j) {
            lower.push_back(&synth[static_cast<std::size_t>(j)]);
        }
        Field3D psi = synth[static_cast<std::size_t>(k)];
        itp.relax_deflated(psi, lower, ses::kH2plusAtlasFlushSteps);
        const double e = ses::mean_energy(psi, rig.v);
        const double e_ref = ses::mean_energy(ref[static_cast<std::size_t>(k)], rig.v);
        const double ov = std::norm(ses::inner_product(ref[static_cast<std::size_t>(k)], psi));
        const double var = energy_variance3(psi, rig.v);
        std::printf("  member %d: E %.5f (grid %.5f), overlap %.5f, Var %.2e\n", k, e,
                    e_ref, ov, var);
        EXPECT_LT(var, 1e-2) << "member " << k;
        EXPECT_NEAR(e, e_ref, k <= 1 ? 1e-3 : 3e-3) << "member " << k;
        EXPECT_GT(ov, k <= 1 ? 0.999 : 0.99) << "member " << k;
    }
}

// A rotated ion flushes in its own V (relax_potential = the live axis).
TEST(AtlasFlush, TiltedAxisSigmaGLandsOnItsOwnGridGround) {
    const AtlasFlushRig rig;
    const Vec3d n = ses::normalized(Vec3d{0.6, 0.1, 0.8});
    const Vec3d e1 = ses::normalized(ses::cross(n, Vec3d{0.0, 1.0, 0.0}));
    const std::vector<double> v = ses::regularized_coulomb_potential(
        rig.g, 1.0, std::vector<Vec3d>{(0.5 * rig.R) * n, (-0.5 * rig.R) * n});
    const ses::ImaginaryTimePropagator3D itp{rig.g, v, ses::kH2plusAtlasFlushDtau};
    Field3D sg = ses::synthesize_h2plus(rig.g, rig.atlas[0], 0, n, e1);
    ses::normalize(sg);
    Field3D ground = sg;
    itp.relax(ground, 400);
    Field3D psi = sg;
    itp.relax(psi, ses::kH2plusAtlasFlushSteps);
    EXPECT_NEAR(ses::mean_energy(psi, v), ses::mean_energy(ground, v), 1e-3);
    EXPECT_GT(std::norm(ses::inner_product(ground, psi)), 0.999);
}

}  // namespace
