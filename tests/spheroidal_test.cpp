// RED: exact prolate-spheroidal H2+ eigensolver (ses.spheroidal).
// Oracle E_elec (electronic, excludes 1/R) at R=2; Scott arXiv:physics/0607081,
// Turbiner arXiv:1401.8009.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <utility>
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
    // R=1.875 = exact baked grid point (2h snap at 256^3/+-40).
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

// The baked member must BE the live member on the grid: the generator's
// resampling convention has to match the loader's reconstruction (END-
// INCLUSIVE NODES over [1, xi_max], xi_max = the grid EDGE), and the sample count
// has to resolve the exponential cusp decay.
TEST(Spheroidal, BakedGroundIsTheLiveGroundOnTheGrid) {
    const double R = 1.875;  // the baked grid point
    const Grid1D ax{-16.0, 16.0, 64};
    const Grid3D g{ax, ax, ax};
    const std::vector<ses::H2plusOrbital> baked = ses::h2plus_atlas_baked(R);
    ASSERT_FALSE(baked.empty());
    const ses::H2plusOrbital live = ses::h2plus_orbital(R, 0, 0, 0);
    ASSERT_TRUE(live.valid);
    const Field3D pb = ses::synthesize_h2plus(g, baked.front(), 0);
    const Field3D pl = ses::synthesize_h2plus(g, live, 0);
    const double overlap = std::norm(ses::inner_product(pb, pl));
    EXPECT_GT(overlap, 0.9999) << "baked vs live 1s sigma_g overlap";
    const std::vector<double> v = ses::regularized_coulomb_potential(
        g, 1.0, {{-R / 2, 0.0, 0.0}, {R / 2, 0.0, 0.0}});
    const double eb = ses::mean_energy(pb, v);
    const double el = ses::mean_energy(pl, v);
    EXPECT_NEAR(eb, el, 1e-3) << "baked " << eb << " live " << el;
}

// The table holds one R: any other R must come back EMPTY, never the
// 1.875 orbitals relabelled (their nuclei would sit at +-0.9375 inside a
// potential whose wells are at +-R/2).
TEST(Spheroidal, BakedAtlasRefusesAnROffItsGrid) {
    const std::vector<ses::H2plusOrbital> on = ses::h2plus_atlas_baked(1.875);
    ASSERT_FALSE(on.empty());
    for (const ses::H2plusOrbital& o : on) {
        EXPECT_EQ(o.R, 1.875);
    }
    EXPECT_TRUE(ses::h2plus_atlas_baked(2.0).empty());
    EXPECT_TRUE(ses::h2plus_atlas_baked(6.25).empty());  // "stretched" preset
}

TEST(Spheroidal, BakedGroundSynthesizesLikeTheLiveSolve) {
    const double R = 1.875;  // the baked grid point
    const Grid1D ax{-16.0, 16.0, 128};
    const Grid3D g{ax, ax, ax};
    const ses::H2plusOrbital b0 = ses::h2plus_atlas_baked(R).front();
    const Field3D psi = ses::synthesize_h2plus(g, b0, 0);
    const std::vector<double> v = ses::regularized_coulomb_potential(
        g, 1.0, {{-R / 2, 0.0, 0.0}, {R / 2, 0.0, 0.0}});
    const double e = ses::mean_energy(psi, v);
    EXPECT_LT(e, -0.5) << "baked ground synthesizes a bound state";
}

// ---- m > 0: the bond axis is a node ----
// For m > 0 the orbital goes like rho^m e^{i m phi} at the axis: M(+-1) = 0,
// Lambda(1) = 0 (as (1 - eta^2)^(m/2), (xi^2 - 1)^(m/2)), and a sampled field
// is EXACTLY zero on a grid line that is the axis, for both partners.

// A linear extrapolation over the edge half-cell does not land on zero: it
// leaves 1pi_u's Lambda(1) at a quarter of its peak and turns the sign of the
// m >= 3 ends.
TEST(Spheroidal, BakedProfilesOfMAboveZeroVanishAtTheirAxisEnds) {
    const std::vector<ses::H2plusOrbital> baked = ses::h2plus_atlas_baked(1.875);
    ASSERT_FALSE(baked.empty());
    int checked = 0;
    for (std::size_t k = 0; k < baked.size(); ++k) {
        const ses::H2plusOrbital& o = baked[k];
        if (o.m == 0) {
            continue;
        }
        ++checked;
        EXPECT_EQ(o.M.front(), 0.0) << "orbital " << k << ", m = " << o.m;
        EXPECT_EQ(o.M.back(), 0.0) << "orbital " << k << ", m = " << o.m;
        EXPECT_EQ(o.lambda.front(), 0.0) << "orbital " << k << ", m = " << o.m;
    }
    EXPECT_GT(checked, 0);
}

// 32^3 over +-8: x = y = 0 is the grid line i = j = 16, so with the nuclei on
// z the bond axis runs through cells; x and y share one axis grid, so the
// grid is symmetric under x <-> y.
struct AxisRig {
    Grid3D g{Grid1D{-8.0, 8.0, 32}, Grid1D{-8.0, 8.0, 32},
             Grid1D{-8.0, 8.0, 32}};
    double R = 1.875;
    Vec3d n{0.0, 0.0, 1.0};
    Vec3d e1{1.0, 0.0, 0.0};
    std::vector<ses::H2plusOrbital> baked = ses::h2plus_atlas_baked(R);
    Field3D synth(const ses::H2plusOrbital& o, int partner) const {
        return ses::synthesize_h2plus(g, o, partner, n, e1);
    }
    // The first baked member with this m.
    const ses::H2plusOrbital& with_m(int m) const {
        for (const ses::H2plusOrbital& o : baked) {
            if (o.m == m) {
                return o;
            }
        }
        return baked.front();
    }
};

double peak_abs(const Field3D& f) {
    double pk = 0.0;
    for (const std::complex<double>& z : f.data()) {
        pk = std::max(pk, std::abs(z));
    }
    return pk;
}

TEST(SynthesizeH2plus, MAboveZeroIsExactlyZeroOnTheBondAxis) {
    const AxisRig rig;
    ASSERT_FALSE(rig.baked.empty());
    std::vector<ses::H2plusOrbital> orbs = {rig.with_m(1), rig.with_m(2),
                                            rig.with_m(3)};
    ASSERT_EQ(orbs[0].m, 1);
    ASSERT_EQ(orbs[1].m, 2);
    ASSERT_EQ(orbs[2].m, 3);
    // ...and the live solve, whose profiles sit on cell centres.
    orbs.push_back(ses::h2plus_orbital(rig.R, 1, 0, 0));
    ASSERT_TRUE(orbs.back().valid);
    for (const ses::H2plusOrbital& o : orbs) {
        for (const int partner : {0, 1}) {
            const Field3D f = rig.synth(o, partner);
            ASSERT_GT(peak_abs(f), 0.0);
            for (int k = 0; k < rig.g.z.n; ++k) {
                EXPECT_EQ(f(16, 16, k), std::complex<double>{})
                    << "m = " << o.m << ", partner " << partner << ", k = "
                    << k;
            }
        }
    }
}

// The two partners are one function turned by pi / (2m) about the axis. On a
// grid symmetric under x <-> y that reads as a transposition:
//   m = 1:  pi_y(x, y) = pi_x(y, x)
//   m = 2:  x^2 - y^2 is odd under the swap, 2xy even
//   m = 3:  Im (x + iy)^3 = -Re (y + ix)^3
// -- and degenerate partners have one energy.
TEST(SynthesizeH2plus, PartnersAreEachOthersTurnAboutTheAxis) {
    const AxisRig rig;
    ASSERT_FALSE(rig.baked.empty());
    // worst |a(i, j, k) - sign * b(j, i, k)| over the grid, relative to peak
    const auto mismatch = [&](const Field3D& a, const Field3D& b, double sign) {
        double worst = 0.0;
        for (int k = 0; k < rig.g.z.n; ++k) {
            for (int j = 0; j < rig.g.y.n; ++j) {
                for (int i = 0; i < rig.g.x.n; ++i) {
                    worst = std::max(worst,
                                     std::abs(a(i, j, k) - sign * b(j, i, k)));
                }
            }
        }
        return worst / peak_abs(a);
    };
    const Field3D p1c = rig.synth(rig.with_m(1), 0);
    const Field3D p1s = rig.synth(rig.with_m(1), 1);
    EXPECT_LT(mismatch(p1s, p1c, +1.0), 1e-12) << "m = 1";
    const Field3D p2c = rig.synth(rig.with_m(2), 0);
    const Field3D p2s = rig.synth(rig.with_m(2), 1);
    EXPECT_LT(mismatch(p2c, p2c, -1.0), 1e-12) << "m = 2, cos";
    EXPECT_LT(mismatch(p2s, p2s, +1.0), 1e-12) << "m = 2, sin";
    const Field3D p3c = rig.synth(rig.with_m(3), 0);
    const Field3D p3s = rig.synth(rig.with_m(3), 1);
    EXPECT_LT(mismatch(p3s, p3c, -1.0), 1e-12) << "m = 3";

    const std::vector<double> v = ses::regularized_coulomb_potential(
        rig.g, 1.0,
        std::vector<Vec3d>{{0.0, 0.0, 0.5 * rig.R}, {0.0, 0.0, -0.5 * rig.R}});
    EXPECT_NEAR(ses::mean_energy(p1c, v), ses::mean_energy(p1s, v), 1e-10);
    EXPECT_NEAR(ses::mean_energy(p3c, v), ses::mean_energy(p3s, v), 1e-10);
}

// The baked member must be the live member for m > 0 as well -- also next to
// the axis, where the profile goes like the square root of the distance from
// its end. Read through the smooth part the two agree to 1 - overlap = 1.8e-6
// and 1.6 % of the peak in the worst cells, the first ones off the
// internuclear segment (xi = 1.05); gates about 1.5 times that. What is left
// there is two resolutions, not a convention: the baked xi nodes are 0.17
// apart, and the live finite-difference solve has its own boundary layer
// (its smooth part runs 17 % and 2.4 % high at its first two centres).
// Interpolating Lambda itself, with the end node a linear extrapolation,
// read 5.6e-5 and 7.0 %.
TEST(Spheroidal, BakedPiUIsTheLivePiUOnTheGrid) {
    const double R = 1.875;
    const Grid1D ax{-10.0, 10.0, 64};  // h = 0.3125 = the scene's
    const Grid3D g{ax, ax, ax};
    const std::vector<ses::H2plusOrbital> baked = ses::h2plus_atlas_baked(R);
    ASSERT_GT(baked.size(), 2u);
    const ses::H2plusOrbital& b = baked[2];
    ASSERT_EQ(b.m, 1);
    ASSERT_EQ(b.n_eta, 0);
    ASSERT_EQ(b.n_xi, 0);
    const ses::H2plusOrbital live = ses::h2plus_orbital(R, 1, 0, 0);
    ASSERT_TRUE(live.valid);
    for (const int partner : {0, 1}) {
        const Field3D pb = ses::synthesize_h2plus(g, b, partner,
                                                  Vec3d{0.0, 0.0, 1.0},
                                                  Vec3d{1.0, 0.0, 0.0});
        const Field3D pl = ses::synthesize_h2plus(g, live, partner,
                                                  Vec3d{0.0, 0.0, 1.0},
                                                  Vec3d{1.0, 0.0, 0.0});
        const double overlap = std::norm(ses::inner_product(pb, pl));
        double worst = 0.0;
        for (std::size_t c = 0; c < pb.data().size(); ++c) {
            worst = std::max(worst, std::abs(pb.data()[c] - pl.data()[c]));
        }
        EXPECT_GT(overlap, 1.0 - 3e-6) << "partner " << partner;
        EXPECT_LT(worst, 2.5e-2 * peak_abs(pl)) << "partner " << partner;
    }
}

// ---- atlas flush (CONTRACT for the H2+ scene's prepare(k)) ----
// A spheroidal orbital SAMPLED on h = 0.3125 is not a grid eigenstate: the
// cusp's high-k content puts 1sigma_g ~11 mHa above the grid ground with
// Var ~0.7 Ha^2 (continuum that disperses over the box). The scene's flush --
// ITP kH2plusAtlasFlushSteps x kH2plusAtlasFlushDtau, deflated against the
// lower SYNTHESIZED members -- must land each member on its grid state:
// |E - E_grid| < 1 mHa, overlap > 0.9999, Var < 1e-2.

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
    // Before: not a grid eigenstate (the defect the flush exists for) -- a
    // gap several times the 1 mHa tolerance and a continuum-sized variance.
    EXPECT_GT(ses::mean_energy(sg, rig.v) - e_grid, 5e-3);
    EXPECT_GT(energy_variance3(sg, rig.v), 0.1);
    Field3D psi = sg;
    itp.relax(psi, ses::kH2plusAtlasFlushSteps);
    const double e = ses::mean_energy(psi, rig.v);
    std::printf("  sigma_g: E %.5f -> %.5f (grid %.5f), Var %.2e, overlap %.5f\n",
                ses::mean_energy(sg, rig.v), e, e_grid, energy_variance3(psi, rig.v),
                std::norm(ses::inner_product(ground, psi)));
    EXPECT_NEAR(e, e_grid, 1e-3);
    EXPECT_GT(std::norm(ses::inner_product(ground, psi)), 0.9999);
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
    EXPECT_GT(ses::mean_energy(su, rig.v) - e_grid, 5e-3);
    EXPECT_GT(energy_variance3(su, rig.v), 0.1);
    // The scene deflates against the synthesized (not flushed) lower members.
    Field3D psi = su;
    itp.relax_deflated(psi, {&sg}, ses::kH2plusAtlasFlushSteps);
    const double e = ses::mean_energy(psi, rig.v);
    std::printf("  sigma_u*: E %.5f -> %.5f (grid %.5f), overlap %.5f\n",
                ses::mean_energy(su, rig.v), e, e_grid,
                std::norm(ses::inner_product(excited, psi)));
    EXPECT_NEAR(e, e_grid, 1e-3);
    EXPECT_GT(std::norm(ses::inner_product(excited, psi)), 0.9999);
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

// The scene exposes up to 24 members. For k >= 2 tau = 2 is not a converged
// relaxation: a sampled member's admixture of SAME-symmetry higher bound
// states decays only as e^{-dE tau} (measured 1 - overlap: 1e-5, 1.3e-4,
// 1.4e-3 for k = 2..4). What the flush promises for every member: the
// continuum is gone (Var < 1e-2), E within 3 mHa of the converged grid
// state, overlap > 0.99 (k <= 1: 1 mHa and the 0.9999 contract; measured
// 1 - overlap < 5e-6).
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
        EXPECT_GT(ov, k <= 1 ? 0.9999 : 0.99) << "member " << k;
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
    // the flush contract (measured 1 - overlap ~1e-8)
    EXPECT_GT(std::norm(ses::inner_product(ground, psi)), 0.9999);
}

}  // namespace
