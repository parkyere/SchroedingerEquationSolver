// RED: ses.rotor -- Ehrenfest rigid LINEAR rotor for the H2+ rotor scene.
// Nuclei = classical unit axis n with angular momentum L PERPENDICULAR to n
// (a linear molecule carries no nuclear angular momentum along its axis;
// that component is the electronic Lambda), moment of inertia I = mu R^2.
// The electron cloud drives it through the orientation torque
//     tau = -dE/d(orientation) = (R/2) n x (F1 - F2),  F_k = <psi|grad V_k|psi>
// (force ON nucleus k from the electron). Sign oracle below is the finite
// difference of E(n) itself, independent of that formula.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <numbers>
#include <vector>
import ses.field;
import ses.grid;
import ses.potential;
import ses.rotor;
import ses.spheroidal;
import ses.vec;
import ses.wavepacket;
import ses.propagator;
import ses.observables;
import ses.h2plus_atlas_loader;

namespace {

using ses::Vec3d;

constexpr double kPi = std::numbers::pi;

TEST(RigidRotor, OmegaEnergyPeriodFollowFromLAndInertia) {
    const ses::RigidRotor r{Vec3d{0.0, 0.0, 1.0}, Vec3d{0.0, 2.0, 0.0}, 4.0};
    const Vec3d w = ses::rotor_omega(r);  // L / I
    EXPECT_NEAR(w.x, 0.0, 1e-15);
    EXPECT_NEAR(w.y, 0.5, 1e-15);
    EXPECT_NEAR(w.z, 0.0, 1e-15);
    EXPECT_NEAR(ses::rotor_energy(r), 0.5, 1e-15);        // L^2 / (2I)
    EXPECT_NEAR(ses::rotor_period(r), 4.0 * kPi, 1e-12);  // 2 pi I / |L|
}

TEST(RigidRotor, KickProjectsOutTheAxisComponent) {
    ses::RigidRotor r{Vec3d{0.0, 0.0, 1.0}, Vec3d{}, 1.0};
    ses::rotor_kick(r, Vec3d{0.0, 0.0, 1.0}, 3.0);  // along the axis: nothing
    EXPECT_NEAR(ses::length(r.L), 0.0, 1e-15);
    ses::rotor_kick(r, Vec3d{1.0, 0.0, 0.0}, 3.0);
    EXPECT_NEAR(r.L.x, 3.0, 1e-15);
    // (0, 1, 1)/sqrt2 scaled by sqrt2 = (0, 1, 1): only the y part survives.
    const double s = 1.0 / std::sqrt(2.0);
    ses::rotor_kick(r, Vec3d{0.0, s, s}, std::sqrt(2.0));
    EXPECT_NEAR(r.L.x, 3.0, 1e-12);
    EXPECT_NEAR(r.L.y, 1.0, 1e-12);
    EXPECT_NEAR(r.L.z, 0.0, 1e-12);
}

TEST(RigidRotor, FreeRotationIsExactForAnyStepSize) {
    // I = 2, L = y-hat: omega = 1/2 about y; z-hat -> (sin wt, 0, cos wt).
    ses::RigidRotor r{Vec3d{0.0, 0.0, 1.0}, Vec3d{0.0, 1.0, 0.0}, 2.0};
    const double dt = 0.37;
    const int n = 50;
    for (int i = 0; i < n; ++i) {
        ses::rotor_step(r, Vec3d{}, dt);
    }
    const double th = 0.5 * dt * n;
    EXPECT_NEAR(r.n.x, std::sin(th), 1e-12);
    EXPECT_NEAR(r.n.y, 0.0, 1e-12);
    EXPECT_NEAR(r.n.z, std::cos(th), 1e-12);
    EXPECT_NEAR(ses::length(r.n), 1.0, 1e-13);
    EXPECT_NEAR(ses::dot(r.L, r.n), 0.0, 1e-13);
    EXPECT_NEAR(ses::length(r.L), 1.0, 1e-13);  // free: L conserved
}

TEST(RigidRotor, ConstantTorqueAcceleratesUniformly) {
    // From rest under tau y-hat: L = tau t, angle = tau t^2 / (2I).
    ses::RigidRotor r{Vec3d{0.0, 0.0, 1.0}, Vec3d{}, 2.0};
    const double tau = 0.1;
    const double dt = 1e-3;
    const int n = 2000;
    for (int i = 0; i < n; ++i) {
        ses::rotor_step(r, Vec3d{0.0, tau, 0.0}, dt);
    }
    const double t = dt * n;
    EXPECT_NEAR(r.L.y, tau * t, 1e-10);
    EXPECT_NEAR(std::atan2(r.n.x, r.n.z), tau * t * t / (2.0 * 2.0), 1e-6);
    EXPECT_NEAR(ses::length(r.n), 1.0, 1e-12);
}

TEST(RigidRotor, TorqueAlongTheAxisCannotSpinALinearMolecule) {
    ses::RigidRotor r{Vec3d{0.0, 0.0, 1.0}, Vec3d{}, 1.0};
    for (int i = 0; i < 100; ++i) {
        ses::rotor_step(r, Vec3d{0.0, 0.0, 5.0}, 0.01);
    }
    EXPECT_NEAR(ses::length(r.L), 0.0, 1e-14);
    EXPECT_NEAR(r.n.z, 1.0, 1e-14);
}

ses::Grid3D cube(double half, int n) {
    return ses::Grid3D{ses::Grid1D{-half, half, n}, ses::Grid1D{-half, half, n},
                       ses::Grid1D{-half, half, n}};
}

// <psi| V(n) |psi> for H2+ nuclei at +-(R/2) n (unit charges).
double orientation_energy(const ses::Field3D& psi, double R, Vec3d n) {
    const ses::Grid3D& g = psi.grid();
    const std::vector<double> v = ses::regularized_coulomb_potential(
        g, 1.0, std::vector<Vec3d>{(0.5 * R) * n, (-0.5 * R) * n});
    double e = 0.0;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                e += std::norm(psi(i, j, k)) *
                     v[static_cast<std::size_t>(g.flat(i, j, k))];
            }
        }
    }
    return e * g.cell_volume();
}

TEST(RotorTorque, AxisSymmetricCloudExertsNone) {
    const ses::Grid3D g = cube(6.0, 48);
    ses::Field3D psi = ses::gaussian_wavepacket(g, Vec3d{}, Vec3d{1.2, 1.2, 1.2},
                                                Vec3d{});
    ses::normalize(psi);
    const Vec3d tau = ses::rotor_torque(psi, 2.0, Vec3d{0.0, 0.0, 1.0});
    EXPECT_NEAR(tau.x, 0.0, 1e-10);
    EXPECT_NEAR(tau.y, 0.0, 1e-10);
    EXPECT_NEAR(tau.z, 0.0, 1e-10);
}

TEST(RotorTorque, PullsTheAxisTowardATiltedCloudWithMinusDEDTheta) {
    // Two lobes along m = R_y(eps) z-hat (tilted toward +x). The torque must
    // rotate n = z-hat toward m: about +y (R_y(+) takes z-hat to +x), with
    // magnitude -dE/dtheta from the energy's own finite difference.
    const ses::Grid3D g = cube(6.0, 48);
    const double eps = 0.3;
    const Vec3d m{std::sin(eps), 0.0, std::cos(eps)};
    const Vec3d sig{0.7, 0.7, 0.7};
    ses::Field3D psi = ses::gaussian_wavepacket(g, 1.0 * m, sig, Vec3d{});
    const ses::Field3D lobe2 = ses::gaussian_wavepacket(g, -1.0 * m, sig, Vec3d{});
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                psi(i, j, k) += lobe2(i, j, k);
            }
        }
    }
    ses::normalize(psi);
    const double R = 2.0;
    const Vec3d tau = ses::rotor_torque(psi, R, Vec3d{0.0, 0.0, 1.0});
    const double d = 1e-3;
    const double e_plus = orientation_energy(psi, R, Vec3d{std::sin(d), 0.0, std::cos(d)});
    const double e_minus = orientation_energy(psi, R, Vec3d{-std::sin(d), 0.0, std::cos(d)});
    const double de_dtheta = (e_plus - e_minus) / (2.0 * d);
    EXPECT_LT(de_dtheta, 0.0) << "energy must fall as n turns toward the cloud";
    EXPECT_GT(tau.y, 0.0);
    EXPECT_NEAR(tau.y, -de_dtheta, 0.03 * std::abs(de_dtheta));
    EXPECT_NEAR(tau.x, 0.0, 1e-8);  // lobes lie in the xz-plane
    EXPECT_NEAR(tau.z, 0.0, 1e-8);
}

// ---- dissociation cap: J is limited to what the REAL molecule can hold ----
// Effective rotational potential V_J(R) = V(R) + J(J+1)/(2 mu R^2). BOUND
// means a well whose floor lies below the dissociation asymptote V(inf)
// (approximated by the last sample of the bare curve); a well held only by
// the centrifugal barrier is quasi-bound (it tunnels away) and does NOT
// count. rotor_j_max = the last bound J. Independent oracle: a plain
// three-point local-minimum scan against that threshold.
bool has_bound_well(const std::vector<double>& r, const std::vector<double>& v,
                    double mu, int j) {
    const double c = j * (j + 1.0) / (2.0 * mu);
    const double asymptote = v.back();
    for (std::size_t i = 1; i + 1 < r.size(); ++i) {
        const double a = v[i - 1] + c / (r[i - 1] * r[i - 1]);
        const double b = v[i] + c / (r[i] * r[i]);
        const double d = v[i + 1] + c / (r[i + 1] * r[i + 1]);
        if (b < a && b < d && b < asymptote) {
            return true;
        }
    }
    return false;
}

TEST(RotorJMax, LastJWithAWellOnAMorseCurve) {
    // Morse with H2+-like numbers: D_e = 0.1026 Ha, R_e = 2.0, a = 0.72; mu = 918.
    const double de = 0.1026;
    const double re = 2.0;
    const double a = 0.72;
    const double mu = 918.0;
    std::vector<double> r;
    std::vector<double> v;
    for (int i = 0; i < 3000; ++i) {
        const double x = 1.0 + 14.0 * i / 2999.0;
        const double u = 1.0 - std::exp(-a * (x - re));
        r.push_back(x);
        v.push_back(de * u * u - de);
    }
    const int jmax = ses::rotor_j_max(r, v, mu);
    EXPECT_GT(jmax, 10);
    EXPECT_TRUE(has_bound_well(r, v, mu, jmax));
    EXPECT_FALSE(has_bound_well(r, v, mu, jmax + 1));
}

TEST(RotorJMax, RealH2plusGroundCurveHoldsAboutThirtyFiveQuanta) {
    // V(R) = E_1s-sigma-g(R) + 1/R from the exact spheroidal solver; mu = m_p/2.
    // The X state's v = 0 rotational ladder ends near J ~ 35 (literature); the
    // classical well floor sits one zero-point energy lower, so the cap lands
    // a few quanta above that.
    std::vector<double> r;
    std::vector<double> v;
    for (int i = 0; i <= 60; ++i) {
        const double R = 1.0 + 0.15 * i;  // 1.0 .. 10.0 bohr
        const ses::H2plusOrbital o = ses::h2plus_orbital(R, 0, 0, 0);
        ASSERT_TRUE(o.valid) << "R = " << R;
        r.push_back(R);
        v.push_back(o.energy + 1.0 / R);
    }
    const int jmax = ses::rotor_j_max(r, v, 918.08);
    std::fprintf(stderr, "H2+ rigid-rotor bound cap: J_max = %d\n", jmax);
    EXPECT_GE(jmax, 30);
    EXPECT_LE(jmax, 42);
}

TEST(RigidRotor, KickRefusesBeyondTheCapButAlwaysAllowsSlowingDown) {
    ses::RigidRotor r{Vec3d{0.0, 0.0, 1.0}, Vec3d{}, 1.0};
    EXPECT_TRUE(ses::rotor_kick(r, Vec3d{1.0, 0.0, 0.0}, 3.0, 5.0));
    EXPECT_FALSE(ses::rotor_kick(r, Vec3d{1.0, 0.0, 0.0}, 3.0, 5.0));  // 6 > 5
    EXPECT_NEAR(r.L.x, 3.0, 1e-15);  // refused kick leaves L untouched
    EXPECT_TRUE(ses::rotor_kick(r, Vec3d{-1.0, 0.0, 0.0}, 3.0, 5.0));
    EXPECT_NEAR(ses::length(r.L), 0.0, 1e-15);
}

// ---- Ehrenfest integration scheme (CONTRACT for the scene's GPU batches):
// inside a step batch the nuclei keep turning (free rotation is exact), so
// the electron must see V(n(t)) at EVERY kick -- half-kick V(n_0), then
// [drift, full-kick V(n_k)], drift, half-kick V(n_N) -- with the torque
// impulse applied once per batch. A batch that FREEZES V for 16 steps biases
// J by +0.2 hbar per quarter turn (measured; the old arc's 35.18); the
// following scheme keeps |dJ| < 0.05 while the electron exchanges up to
// ~0.1 hbar mid-turn and hands it back (libration, not drift).

struct QuarterTurnResult {
    double j_end;
    double n_y;
};

QuarterTurnResult quarter_turn(bool follow, int batch) {
    const ses::Grid3D g = cube(5.0, 32);  // h = 0.3125 like the 256^3 scene
    const double R = 1.875;
    const double mu = 918.076;
    const double dt = 0.04;
    const ses::H2plusOrbital orb = ses::h2plus_atlas_baked(R).front();
    const Vec3d z{0.0, 0.0, 1.0};
    auto potential = [&](Vec3d n) {
        return ses::regularized_coulomb_potential(
            g, 1.0, std::vector<Vec3d>{(0.5 * R) * n, (-0.5 * R) * n});
    };
    ses::Field3D psi = ses::synthesize_h2plus(g, orb, 0, z, Vec3d{1.0, 0.0, 0.0});
    ses::normalize(psi);
    ses::RigidRotor r{z, Vec3d{35.0, 0.0, 0.0}, mu * R * R};
    const double t_end = 0.25 * ses::rotor_period(r);
    const ses::SplitOperator3D prop{g, potential(z), dt};  // drift tables
    double t = 0.0;
    while (t < t_end) {
        ses::RigidRotor pred = r;
        if (follow) {
            prop.kick(psi, potential(pred.n), 0.5 * dt);
            for (int s = 0; s < batch; ++s) {
                prop.drift(psi);
                ses::rotor_step(pred, Vec3d{}, dt);
                prop.kick(psi, potential(pred.n), s + 1 < batch ? dt : 0.5 * dt);
            }
        } else {
            const ses::SplitOperator3D frozen{g, potential(r.n), dt};
            frozen.step(psi, batch);
            ses::rotor_step(pred, Vec3d{}, batch * dt);
        }
        t += batch * dt;
        const Vec3d tau = ses::rotor_torque(psi, R, pred.n);
        ses::rotor_step(r, tau, batch * dt);
    }
    return {ses::length(r.L), r.n.y};
}

TEST(RotorEhrenfest, PotentialFollowingBatchesConserveJOverAQuarterTurn) {
    const QuarterTurnResult f = quarter_turn(true, 16);
    std::printf("  following B=16: J = %.4f, n_y = %.4f\n", f.j_end, f.n_y);
    EXPECT_LT(f.n_y, -0.99);
    EXPECT_NEAR(f.j_end, 35.0, 0.05);
}

TEST(RotorEhrenfest, FrozenPotentialBatchesBiasJ) {
    const QuarterTurnResult z = quarter_turn(false, 32);
    std::printf("  frozen B=32: J = %.4f, n_y = %.4f\n", z.j_end, z.n_y);
    // Magnitude only: the sign follows where the torque is sampled (batch
    // start +0.2, batch end -0.9 at 32 steps). The scene must carry neither.
    EXPECT_GT(std::abs(z.j_end - 35.0), 0.15);
}

}  // namespace
