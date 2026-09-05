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

}  // namespace
