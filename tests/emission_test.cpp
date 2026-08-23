// RED: semiclassical emission (Ehrenfest + Larmor).
// Ehrenfest: dipole accel a = <grad V> = -<r_ddot>, fed to larmor_power.
// Larmor P = (2/3) alpha^3 |a|^2 with alpha^3 = 1/c^3 (a.u.).
// Coherent superposition emission, =0 for an eigenstate (static density);
// eigenstate spontaneous decay is QED, handled by the Einstein-A jumps.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <vector>
import ses.observables;
import ses.grid;
import ses.vec;
import ses.decay;
import ses.field;
import ses.wavepacket;
import ses.potential;
import ses.emission;


namespace {

using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

Grid3D cube(double half, int n) {
    const Grid1D a{-half, half, n};
    return Grid3D{a, a, a};
}

TEST(MeanPotentialGradient, HarmonicGivesOmegaSquaredMeanPosition) {
    const Grid3D g = cube(12.0, 64);
    const double w = 1.3;
    const std::vector<double> v = ses::harmonic_potential(g, w, Vec3d{});
    const ses::Field3D psi = ses::gaussian_wavepacket(
        g, Vec3d{2.0, 1.0, -0.5}, Vec3d{1.4, 1.4, 1.4}, Vec3d{});
    const Vec3d grad = ses::mean_potential_gradient(psi, v, g);
    const Vec3d r = ses::mean_position(psi);
    EXPECT_NEAR(grad.x, w * w * r.x, 1e-3);
    EXPECT_NEAR(grad.y, w * w * r.y, 1e-3);
    EXPECT_NEAR(grad.z, w * w * r.z, 1e-3);
}

TEST(MeanPotentialGradient, FreeParticleIsZero) {
    const Grid3D g = cube(10.0, 32);
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const ses::Field3D psi = ses::gaussian_wavepacket(
        g, Vec3d{1.0, 0.0, 0.0}, Vec3d{1.5, 1.5, 1.5}, Vec3d{0.0, 0.4, 0.0});
    const Vec3d grad = ses::mean_potential_gradient(psi, v, g);
    EXPECT_NEAR(grad.x, 0.0, 1e-12);
    EXPECT_NEAR(grad.y, 0.0, 1e-12);
    EXPECT_NEAR(grad.z, 0.0, 1e-12);
}

TEST(MeanPotentialGradient, SymmetricCloudInCentralPotentialIsZero) {
    const Grid3D g = cube(12.0, 48);
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, Vec3d{});
    const ses::Field3D psi =
        ses::gaussian_wavepacket(g, Vec3d{}, Vec3d{2.0, 2.0, 2.0}, Vec3d{});
    const Vec3d grad = ses::mean_potential_gradient(psi, v, g);
    EXPECT_NEAR(grad.x, 0.0, 1e-4);
    EXPECT_NEAR(grad.y, 0.0, 1e-4);
    EXPECT_NEAR(grad.z, 0.0, 1e-4);
}

TEST(LarmorPower, ExactFactorAndQuadraticScaling) {
    const double a3 = std::pow(ses::kFineStructureConstant, 3.0);
    EXPECT_DOUBLE_EQ(ses::larmor_power(Vec3d{2.0, 0.0, 0.0}),
                     (2.0 / 3.0) * a3 * 4.0);
    // isotropic: only |a|^2 matters.
    EXPECT_DOUBLE_EQ(ses::larmor_power(Vec3d{1.0, 2.0, 2.0}),
                     (2.0 / 3.0) * a3 * 9.0);
    EXPECT_DOUBLE_EQ(ses::larmor_power(Vec3d{2.0, 0.0, 0.0}),
                     4.0 * ses::larmor_power(Vec3d{1.0, 0.0, 0.0}));
}

// QED photon record: detecting the E1 photon as a plane wave (n, lambda)
// projects the atom onto c_m ∝ conj(e_lambda(n)).D_m -- angular momentum
// conservation lives in this coupling.

using ses::DipoleMatrixElement;
const double kS2 = 1.0 / std::sqrt(2.0);

// sigma+ / pi / sigma- dipole vectors (z quantization axis).
DipoleMatrixElement dip_sigma_plus() {
    return DipoleMatrixElement{{kS2, 0.0}, {0.0, kS2}, {0.0, 0.0}};
}
DipoleMatrixElement dip_pi() {
    return DipoleMatrixElement{{0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
}
DipoleMatrixElement dip_sigma_minus() {
    return DipoleMatrixElement{{kS2, 0.0}, {0.0, -kS2}, {0.0, 0.0}};
}

std::complex<double> cdot(const DipoleMatrixElement& a,
                          const DipoleMatrixElement& b) {
    return std::conj(a.x) * b.x + std::conj(a.y) * b.y + std::conj(a.z) * b.z;
}

TEST(HelicityVector, ConventionAlongZAndTransversality) {
    const DipoleMatrixElement ep = ses::helicity_vector(Vec3d{0.0, 0.0, 1.0}, +1);
    // e_+(z) = (x + iy)/sqrt2.
    EXPECT_NEAR(ep.x.real(), kS2, 1e-12);
    EXPECT_NEAR(ep.x.imag(), 0.0, 1e-12);
    EXPECT_NEAR(ep.y.real(), 0.0, 1e-12);
    EXPECT_NEAR(ep.y.imag(), kS2, 1e-12);
    EXPECT_NEAR(std::abs(ep.z), 0.0, 1e-12);
    const Vec3d dirs[] = {Vec3d{1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0},
                          Vec3d{-kS2, kS2, 0.0}, Vec3d{0.0, 0.0, -1.0}};
    for (const Vec3d& n : dirs) {
        const DipoleMatrixElement p = ses::helicity_vector(n, +1);
        const DipoleMatrixElement m = ses::helicity_vector(n, -1);
        EXPECT_NEAR(std::abs(cdot(p, p)), 1.0, 1e-12);  // unit
        EXPECT_NEAR(std::abs(cdot(m, m)), 1.0, 1e-12);
        EXPECT_NEAR(std::abs(n.x * p.x + n.y * p.y + n.z * p.z), 0.0,
                    1e-12);  // transverse
        EXPECT_NEAR(std::abs(n.x * m.x + n.y * m.y + n.z * m.z), 0.0, 1e-12);
        EXPECT_NEAR(std::abs(cdot(p, m)), 0.0, 1e-12);  // conj-orthogonal
    }
}

TEST(ConditionedAmplitudes, AxialDetectionEnforcesDeltaM) {
    const std::vector<DipoleMatrixElement> dip{dip_sigma_plus(), dip_pi(),
                                               dip_sigma_minus()};
    // Along +z with lambda=+1 only the sigma+ component survives (Delta m = -1).
    const auto cp = ses::conditioned_amplitudes(dip, Vec3d{0.0, 0.0, 1.0}, +1);
    ASSERT_EQ(cp.size(), 3u);
    EXPECT_NEAR(std::abs(cp[0]), 1.0, 1e-12);
    EXPECT_NEAR(std::abs(cp[1]), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(cp[2]), 0.0, 1e-12);
    // lambda=-1 picks the sigma- partner instead.
    const auto cm = ses::conditioned_amplitudes(dip, Vec3d{0.0, 0.0, 1.0}, -1);
    ASSERT_EQ(cm.size(), 3u);
    EXPECT_NEAR(std::abs(cm[0]), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(cm[2]), 1.0, 1e-12);
}

TEST(ConditionedAmplitudes, ObliqueDetectionIsTheExactSuperposition) {
    // n = x, lambda=+1: e_+ = (-z + iy)/sqrt2 -> |c| = (1/2, 1/sqrt2, 1/2).
    const std::vector<DipoleMatrixElement> dip{dip_sigma_plus(), dip_pi(),
                                               dip_sigma_minus()};
    const auto c = ses::conditioned_amplitudes(dip, Vec3d{1.0, 0.0, 0.0}, +1);
    ASSERT_EQ(c.size(), 3u);
    EXPECT_NEAR(std::abs(c[0]), 0.5, 1e-12);
    EXPECT_NEAR(std::abs(c[1]), kS2, 1e-12);
    EXPECT_NEAR(std::abs(c[2]), 0.5, 1e-12);
    double n2 = 0.0;
    for (const auto& z : c) {
        n2 += std::norm(z);
    }
    EXPECT_NEAR(n2, 1.0, 1e-12);
}

TEST(ConditionedAmplitudes, HelicityFixesThePhaseForAngularMomentum) {
    // 3d_z2 -> 2p (tesseral D: -x, -y, +2z up to scale). Same n = +z, either
    // helicity: c_y = -i lam c_x, so <L_z> = 2 Im(c_x* c_y) = -lam — the atom
    // loses exactly the +lam hbar the photon carries.
    const std::vector<DipoleMatrixElement> dip{
        DipoleMatrixElement{{-1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        DipoleMatrixElement{{0.0, 0.0}, {-1.0, 0.0}, {0.0, 0.0}},
        DipoleMatrixElement{{0.0, 0.0}, {0.0, 0.0}, {2.0, 0.0}}};
    for (const int lam : {+1, -1}) {
        const auto c =
            ses::conditioned_amplitudes(dip, Vec3d{0.0, 0.0, 1.0}, lam);
        ASSERT_EQ(c.size(), 3u);
        EXPECT_NEAR(std::abs(c[2]), 0.0, 1e-12);          // pi silent on-axis
        EXPECT_NEAR(std::abs(c[0]), std::abs(c[1]), 1e-12);
        const double lz = 2.0 * std::imag(std::conj(c[0]) * c[1]);
        EXPECT_NEAR(lz, -lam, 1e-12);
    }
}

TEST(ConditionedAmplitudes, AllZeroDipolesStayZero) {
    const std::vector<DipoleMatrixElement> dip{DipoleMatrixElement{},
                                               DipoleMatrixElement{}};
    const auto c = ses::conditioned_amplitudes(dip, Vec3d{0.0, 0.0, 1.0}, +1);
    ASSERT_EQ(c.size(), 2u);
    EXPECT_NEAR(std::abs(c[0]), 0.0, 1e-15);
    EXPECT_NEAR(std::abs(c[1]), 0.0, 1e-15);
}

TEST(PhotonSampling, PiDipoleGivesSinSquaredPatternAndBalancedHelicity) {
    const std::vector<DipoleMatrixElement> dip{dip_pi()};
    std::mt19937 rng(20260823u);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    auto u01 = [&] { return uni(rng); };
    const int n = 20000;
    double cos2 = 0.0;
    double hel = 0.0;
    for (int t = 0; t < n; ++t) {
        const ses::PhotonRecord r = ses::sample_photon_emission(dip, u01);
        const double nrm =
            r.n.x * r.n.x + r.n.y * r.n.y + r.n.z * r.n.z;
        ASSERT_NEAR(nrm, 1.0, 1e-9);
        cos2 += r.n.z * r.n.z;
        hel += r.helicity;
    }
    // sin^2(theta) pattern: <cos^2 theta> = 1/5; linear light = 50/50 helicity.
    EXPECT_NEAR(cos2 / n, 0.2, 0.02);
    EXPECT_NEAR(hel / n, 0.0, 0.03);
}

// Shell dipole vectors: signed tesseral angular factors per destination
// sublevel; radial factor omitted (cancels in conditioning/sampling).

TEST(ShellDipoleVectors, TwoPzToOneS) {
    const auto d = ses::shell_dipole_vectors(1, 0, 0, {0});
    ASSERT_EQ(d.size(), 1u);
    EXPECT_NEAR(d[0].z.real(), 1.0 / std::sqrt(3.0), 1e-12);
    EXPECT_NEAR(std::abs(d[0].x) + std::abs(d[0].y), 0.0, 1e-12);
}

TEST(ShellDipoleVectors, ThreeDz2ToTwoPShellHasTheSinDownSigns) {
    // 3d_z2 -> (2p_z, 2p_x, 2p_y): D = (2z, -x, -y)/sqrt(15).
    const auto d = ses::shell_dipole_vectors(2, 0, 1, {0, 1, -1});
    ASSERT_EQ(d.size(), 3u);
    const double s15 = 1.0 / std::sqrt(15.0);
    EXPECT_NEAR(d[0].z.real(), 2.0 * s15, 1e-12);
    EXPECT_NEAR(d[1].x.real(), -s15, 1e-12);
    EXPECT_NEAR(d[2].y.real(), -s15, 1e-12);
    double sum = 0.0;
    for (const auto& v : d) {
        sum += std::norm(v.x) + std::norm(v.y) + std::norm(v.z);
    }
    EXPECT_NEAR(sum, 2.0 / 5.0, 1e-12);  // tesseral_e1_sq shell sum
}

TEST(ShellDipoleVectors, ForbiddenShellIsAllZero) {
    const auto d = ses::shell_dipole_vectors(1, 0, 1, {0, 1, -1});
    ASSERT_EQ(d.size(), 3u);
    for (const auto& v : d) {
        EXPECT_NEAR(std::abs(v.x) + std::abs(v.y) + std::abs(v.z), 0.0,
                    1e-15);
    }
}

// Photon streak: helix e^{i lambda k s} in the (theta_hat, phi_hat) frame,
// comoving phase (crests ride with the head), ~2 s wall flight, late fade.

TEST(PhotonStreak, FlightConstantsMatchTheUserSpec) {
    EXPECT_EQ(ses::kPhotonFlightTicks, 120);  // ~2 s at 60 fps
    // Must exit the +-80 box even along the corner diagonal.
    EXPECT_GE(ses::kPhotonTravel, std::sqrt(3.0) * 80.0);
}

TEST(PhotonStreak, WavelengthIsInverseInEnergy) {
    EXPECT_DOUBLE_EQ(ses::photon_display_wavelength(0.375), 25.0);
    EXPECT_DOUBLE_EQ(ses::photon_display_wavelength(0.75), 12.5);
}

TEST(PhotonStreak, AlphaHoldsThenFadesToZero) {
    EXPECT_DOUBLE_EQ(ses::photon_streak_alpha(0.0), 1.0);
    EXPECT_DOUBLE_EQ(ses::photon_streak_alpha(0.5), 1.0);
    EXPECT_LT(ses::photon_streak_alpha(0.95), ses::photon_streak_alpha(0.75));
    EXPECT_DOUBLE_EQ(ses::photon_streak_alpha(1.0), 0.0);
    double prev = 1.0;
    for (int i = 0; i <= 20; ++i) {
        const double a = ses::photon_streak_alpha(i / 20.0);
        EXPECT_LE(a, prev + 1e-12);
        prev = a;
    }
}

TEST(PhotonStreak, HelixTwistSenseFollowsHelicity) {
    const double de = 0.375;  // lambda_display = 25 Bohr
    const double k = 2.0 * 3.14159265358979323846 / 25.0;
    for (const int lam : {+1, -1}) {
        const ses::PhotonRecord ph{Vec3d{0.0, 0.0, 1.0}, lam};
        const auto v = ses::photon_streak_vertices(ph, de, 0.5);
        ASSERT_EQ(v.size(),
                  static_cast<std::size_t>(ses::kPhotonStreakPoints));
        const double sh = 0.5 * ses::kPhotonTravel;
        const std::size_t body = v.size() - 1;
        double prev_s = -1.0;
        for (std::size_t i = 0; i < body; ++i) {
            const double s = v[i].z;  // n = z: axial coordinate
            EXPECT_GT(s, prev_s);
            prev_s = s;
            const double tr =
                std::sqrt(v[i].x * v[i].x + v[i].y * v[i].y);
            EXPECT_NEAR(tr, ses::kPhotonStreakRadius, 1e-9);
        }
        // Comoving phase: at the head s = sh the transverse is +R x_hat...
        const Vec3d& head = v[body - 1];
        EXPECT_NEAR(head.z, sh, 1e-9);
        EXPECT_NEAR(head.x, ses::kPhotonStreakRadius, 1e-9);
        EXPECT_NEAR(head.y, 0.0, 1e-9);
        // ...and a quarter display-wavelength behind it sits at -lambda R y_hat.
        for (std::size_t i = 0; i < body; ++i) {
            if (std::abs(v[i].z - (sh - 25.0 / 4.0)) < 1e-6) {
                EXPECT_NEAR(v[i].x, 0.0, 1e-9);
                EXPECT_NEAR(v[i].y, -lam * ses::kPhotonStreakRadius, 1e-9);
            }
        }
        // Twist sense: consecutive transverse vectors rotate with sign lam.
        for (std::size_t i = 0; i + 1 < body; ++i) {
            const double cross = v[i].x * v[i + 1].y - v[i].y * v[i + 1].x;
            EXPECT_GT(lam * cross, 0.0);
        }
        (void)k;
    }
}

TEST(PhotonStreak, TipIsOnAxisAheadAndTailAnchorsAtTheNucleus) {
    const ses::PhotonRecord ph{Vec3d{0.0, 0.0, 1.0}, +1};
    // Early flight: the tail is still anchored at the nucleus.
    const auto early = ses::photon_streak_vertices(ph, 0.375, 0.1);
    ASSERT_FALSE(early.empty());
    EXPECT_NEAR(early.front().z, 0.0, 1e-9);
    // The tip is on-axis, strictly ahead of every body vertex.
    const Vec3d tip = early.back();
    EXPECT_NEAR(std::sqrt(tip.x * tip.x + tip.y * tip.y), 0.0, 1e-9);
    for (std::size_t i = 0; i + 1 < early.size(); ++i) {
        EXPECT_LT(early[i].z, tip.z);
    }
    // Full flight reaches past the box.
    const auto done = ses::photon_streak_vertices(ph, 0.375, 1.0);
    ASSERT_FALSE(done.empty());
    EXPECT_GE(done.back().z, ses::kPhotonTravel);
}

TEST(PhotonSampling, CircularDipoleCorrelatesHelicityWithHemisphere) {
    const std::vector<DipoleMatrixElement> dip{dip_sigma_plus()};
    std::mt19937 rng(9781u);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    auto u01 = [&] { return uni(rng); };
    const int n = 20000;
    double cos2 = 0.0;
    double lam_cos = 0.0;
    for (int t = 0; t < n; ++t) {
        const ses::PhotonRecord r = ses::sample_photon_emission(dip, u01);
        cos2 += r.n.z * r.n.z;
        lam_cos += r.helicity * r.n.z;
    }
    // (1 + cos^2)/2 pattern: <cos^2> = 2/5; sigma+ helicity rides +z:
    // <lambda cos theta> = 1/2.
    EXPECT_NEAR(cos2 / n, 0.4, 0.02);
    EXPECT_NEAR(lam_cos / n, 0.5, 0.02);
}

}  // namespace
