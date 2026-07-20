// RED: 1D harmonic-oscillator ladder operators over Field1D (atomic units,
// m = hbar = 1):
//     a    = sqrt(omega/2) x + 1/sqrt(2 omega) d/dx
//     adag = sqrt(omega/2) x - 1/sqrt(2 omega) d/dx
// Derivative is spectral (FFT), matching the split-operator periodic grid.
// gaussian_wavepacket at sigma = 1/sqrt(2 omega) IS the exact HO ground state,
// so the whole Fock chain is reachable by repeated raises, no diagonalization.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

import ses.field;
import ses.grid;
import ses.ladder;
import ses.observables;
import ses.potential;
import ses.wavepacket;

namespace {

constexpr double kOmega = 0.25;
// Nyquist matched to the band, NOT oversampled: the spectral derivative
// amplifies round-off by k_max/sqrt(2 omega) per raise, so more points walk
// the noise floor UP the chain. 256/+-20 stays clean to n=8; 1024 loses it.
const ses::Grid1D kGrid{-20.0, 20.0, 256};

ses::Field1D ho_ground() {
    const double sigma = 1.0 / std::sqrt(2.0 * kOmega);
    return ses::gaussian_wavepacket(kGrid, 0.0, sigma, 0.0);
}

double overlap_sq(const ses::Field1D& a, const ses::Field1D& b) {
    std::complex<double> acc{};
    for (int i = 0; i < a.size(); ++i) {
        acc += std::conj(a[i]) * b[i];
    }
    acc *= a.grid().spacing();
    return std::norm(acc);
}

// Count interior sign changes of a real field (the eigenstate node count).
int node_count(const ses::Field1D& f) {
    int nodes = 0;
    double prev = 0.0;
    const double tiny = 1e-6;
    for (int i = 0; i < f.size(); ++i) {
        const double re = f[i].real();
        if (std::abs(re) < tiny) {
            continue;  // near-zero sample: noise, not a real sign change
        }
        if (prev != 0.0 && (re > 0.0) != (prev > 0.0)) {
            ++nodes;
        }
        prev = re;
    }
    return nodes;
}

// RED: ho_eigenstate -- exact HO eigenstate via Hermite-Gauss recurrence in
// x-space (no derivative, no spectral round-off): the oracle the ladder chain
// is measured against.
TEST(HoEigenstate, IsNormalizedWithExactEnergyAndZeroVariance) {
    const std::vector<double> v = ses::harmonic_potential(kGrid, kOmega);
    for (int n = 0; n <= 12; ++n) {
        const ses::Field1D psi = ses::ho_eigenstate(kGrid, kOmega, n);
        EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-10) << "norm at n = " << n;
        EXPECT_NEAR(ses::mean_energy(psi, v), (n + 0.5) * kOmega, 1e-6)
            << "energy at n = " << n;
        EXPECT_LT(ses::energy_variance(psi, v), 1e-8) << "variance at n = " << n;
    }
}

TEST(HoEigenstate, HasNNodes) {
    for (int n = 0; n <= 8; ++n) {
        EXPECT_EQ(node_count(ses::ho_eigenstate(kGrid, kOmega, n)), n)
            << "node count at n = " << n;
    }
}

TEST(HoEigenstate, IsOrthonormalAcrossLevels) {
    for (int m = 0; m <= 6; ++m) {
        for (int n = 0; n <= 6; ++n) {
            const double ov = overlap_sq(ses::ho_eigenstate(kGrid, kOmega, m),
                                         ses::ho_eigenstate(kGrid, kOmega, n));
            EXPECT_NEAR(ov, m == n ? 1.0 : 0.0, 1e-9)
                << "<" << m << "|" << n << ">^2";
        }
    }
}

TEST(HoEigenstate, MatchesTheCleanLadderChainAtLowLevels) {
    // n <= 4: ladder still clean, so oracle and ladder state match up to a
    // global phase (overlap^2 = 1).
    ses::Field1D psi = ho_ground();
    for (int n = 1; n <= 4; ++n) {
        ses::ladder_raise(psi, kOmega);
        EXPECT_NEAR(overlap_sq(psi, ses::ho_eigenstate(kGrid, kOmega, n)), 1.0,
                    1e-7)
            << "ladder vs oracle at n = " << n;
    }
}

TEST(HoEigenstate, IsOmegaGeneric) {
    const double w = 0.75;
    const std::vector<double> v = ses::harmonic_potential(kGrid, w);
    for (int n = 0; n <= 6; ++n) {
        const ses::Field1D psi = ses::ho_eigenstate(kGrid, w, n);
        EXPECT_NEAR(ses::mean_energy(psi, v), (n + 0.5) * w, 1e-6)
            << "energy at n = " << n;
        EXPECT_LT(ses::energy_variance(psi, v), 1e-8);
    }
}

// RED: ladder_rung_stable(psi, omega, n_from, up) -- for a state KNOWN to be
// |n_from> up to a global phase: raw operator supplies norm^2 + phase, state
// rebuilt from the Hermite oracle so round-off resets each rung instead of
// compounding. Ceiling = grid representability, not the raw-chain noise cap.
TEST(LadderRungStable, RoundTripsFarBeyondTheRawChain) {
    // 25 rungs up/down, far past the raw-chain cap (~12 at omega = 0.25).
    const std::vector<double> v = ses::harmonic_potential(kGrid, kOmega);
    ses::Field1D psi = ses::ho_eigenstate(kGrid, kOmega, 0);
    for (int n = 0; n < 25; ++n) {
        const double norm2 = ses::ladder_rung_stable(psi, kOmega, n, true);
        EXPECT_NEAR(norm2, n + 1.0, 1e-5) << "up from n = " << n;
        EXPECT_NEAR(overlap_sq(psi, ses::ho_eigenstate(kGrid, kOmega, n + 1)),
                    1.0, 1e-10)
            << "clean at n = " << n + 1;
    }
    for (int n = 25; n > 0; --n) {
        const double norm2 = ses::ladder_rung_stable(psi, kOmega, n, false);
        EXPECT_NEAR(norm2, static_cast<double>(n), 1e-5) << "down from n = " << n;
    }
    EXPECT_NEAR(overlap_sq(psi, ses::ho_eigenstate(kGrid, kOmega, 0)), 1.0,
                1e-10);
    EXPECT_NEAR(ses::mean_energy(psi, v), 0.5 * kOmega, 1e-9);
}

TEST(LadderRungStable, CarriesTheGlobalPhase) {
    // e^{i theta}|5> must rung to e^{i theta}|6>: the oracle rebuild must
    // carry the phase, not reset it.
    const double theta = 0.7;
    const std::complex<double> ph{std::cos(theta), std::sin(theta)};
    ses::Field1D psi = ses::ho_eigenstate(kGrid, kOmega, 5);
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] *= ph;
    }
    const double norm2 = ses::ladder_rung_stable(psi, kOmega, 5, true);
    EXPECT_NEAR(norm2, 6.0, 1e-6);
    const ses::Field1D oracle = ses::ho_eigenstate(kGrid, kOmega, 6);
    std::complex<double> ov{};
    for (int i = 0; i < psi.size(); ++i) {
        ov += std::conj(oracle[i]) * psi[i];
    }
    ov *= kGrid.spacing();
    EXPECT_NEAR(ov.real(), std::cos(theta), 1e-10);
    EXPECT_NEAR(ov.imag(), std::sin(theta), 1e-10);
}

TEST(LadderRungStable, GroundAnnihilationStillRefuses) {
    ses::Field1D psi = ses::ho_eigenstate(kGrid, kOmega, 0);
    const ses::Field1D before = psi;
    const double norm2 = ses::ladder_rung_stable(psi, kOmega, 0, false);
    EXPECT_LT(norm2, 1e-9);
    for (int i = 0; i < psi.size(); ++i) {
        EXPECT_EQ(psi[i], before[i]) << "annihilated rung must not write psi";
    }
}

// RED: ho_level_cap(grid, omega) -- largest level whose Hermite oracle is
// still faithful (discrete energy within 0.1% of (n+1/2)w). Box-limited at
// soft omega, Nyquist-limited at stiff omega, far above the raw cap.
TEST(HoLevelCap, SitsFarAboveTheRawChainCapAndPeaksNearMatchedNyquist) {
    for (double w : {0.05, 0.25, 1.0, 4.0}) {
        const int level = ses::ho_level_cap(kGrid, w);
        const int raw = ses::ladder_cap(kGrid, w);
        std::fprintf(stderr, "ho_level_cap(w=%.2f) = %d (raw chain %d)\n", w,
                     level, raw);
        EXPECT_GE(level, raw) << "omega=" << w;
    }
    const int c025 = ses::ho_level_cap(kGrid, 0.25);
    const int c1 = ses::ho_level_cap(kGrid, 1.0);
    const int c4 = ses::ho_level_cap(kGrid, 4.0);
    EXPECT_GE(c025, 25);  // box-limited but far beyond raw cap 12
    EXPECT_GE(c1, 80);    // matched Nyquist: the ceiling peaks here
    EXPECT_GT(c1, c025);
    EXPECT_GT(c1, c4);    // stiff well: k-band-limited again
    EXPECT_GE(c4, 25);
}

TEST(HoLevelCap, EveryLevelBelowTheCapIsActuallyClean) {
    // oracle at cap/2 and at the cap is still energy-faithful.
    const double w = 0.25;
    const std::vector<double> v = ses::harmonic_potential(kGrid, w);
    const int cap = ses::ho_level_cap(kGrid, w);
    for (int n : {cap / 2, cap}) {
        const ses::Field1D psi = ses::ho_eigenstate(kGrid, w, n);
        const double e_exact = (n + 0.5) * w;
        EXPECT_NEAR(ses::mean_energy(psi, v), e_exact, 1e-3 * e_exact)
            << "n = " << n;
    }
}

// RED: ladder_fock(psi, omega, up, n_top, &residual) -- superposition rung:
// project onto the truncated Fock basis |0..n_top>, act on coefficients
// (adag: c'_{n+1} = sqrt(n+1) c_n; a: c'_n = sqrt(n+1) c_{n+1}), resynthesize
// from oracles. No spectral derivative, so it works at ANY grid k_max.
// *residual = input's outside-band weight; the caller gates its fallback on it.
TEST(LadderFock, AgreesWithTheRawOperatorOnTheCoarseGrid) {
    ses::Field1D psi{kGrid};
    const ses::Field1D e0 = ses::ho_eigenstate(kGrid, kOmega, 0);
    const ses::Field1D e1 = ses::ho_eigenstate(kGrid, kOmega, 1);
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = e0[i] + e1[i];
    }
    ses::normalize(psi);
    ses::Field1D raw = psi;
    const double n2_raw = ses::ladder_lower(raw, kOmega);
    double residual = 1.0;
    const double n2 = ses::ladder_fock(psi, kOmega, false, 8, &residual);
    EXPECT_NEAR(n2, n2_raw, 1e-9);
    EXPECT_NEAR(n2, 0.5, 1e-9);
    EXPECT_LT(residual, 1e-10);
    EXPECT_NEAR(overlap_sq(psi, raw), 1.0, 1e-9);
    EXPECT_NEAR(overlap_sq(psi, e0), 1.0, 1e-9);
}

TEST(LadderFock, RaisesASuperpositionOnAFineGridWhereTheRawChainDies) {
    // 4096/+-20: k_max ~ 640, raw-chain gain ~900 per rung (raw useless here).
    const ses::Grid1D fine{-20.0, 20.0, 4096};
    ses::Field1D psi{fine};
    const ses::Field1D e2 = ses::ho_eigenstate(fine, kOmega, 2);
    const ses::Field1D e4 = ses::ho_eigenstate(fine, kOmega, 4);
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = e2[i] + e4[i];
    }
    ses::normalize(psi);
    // First raise: counting norm^2 = sum (n+1)|c_n|^2 = (3 + 5)/2 = 4.
    double residual = 1.0;
    EXPECT_NEAR(ses::ladder_fock(psi, kOmega, true, 16, &residual), 4.0, 1e-9);
    EXPECT_LT(residual, 1e-10);
    for (int k = 0; k < 5; ++k) {
        ses::ladder_fock(psi, kOmega, true, 16, &residual);
    }
    // After 6 raises: (adag)^6 (|2> + |4>) ~ sqrt(8!/2!)|8> + sqrt(10!/4!)|10>.
    const double a8 = std::sqrt(40320.0 / 2.0);       // 8!/2!
    const double a10 = std::sqrt(3628800.0 / 24.0);   // 10!/4!
    const ses::Field1D e8 = ses::ho_eigenstate(fine, kOmega, 8);
    const ses::Field1D e10 = ses::ho_eigenstate(fine, kOmega, 10);
    ses::Field1D expect{fine};
    for (int i = 0; i < expect.size(); ++i) {
        expect[i] = a8 * e8[i] + a10 * e10[i];
    }
    ses::normalize(expect);
    EXPECT_NEAR(overlap_sq(psi, expect), 1.0, 1e-9);
}

TEST(LadderFock, ReportsOutsideBandResidualAndRefusesNothingItself) {
    // |10> against a band capped at n_top = 5: all residual, state untouched
    // (the caller's fallback signal).
    ses::Field1D psi = ses::ho_eigenstate(kGrid, kOmega, 10);
    const ses::Field1D before = psi;
    double residual = 0.0;
    ses::ladder_fock(psi, kOmega, true, 5, &residual);
    EXPECT_GT(residual, 0.99);
    for (int i = 0; i < psi.size(); ++i) {
        EXPECT_EQ(psi[i], before[i]);
    }
}

TEST(LadderFock, AnnihilatesThePureGroundOnLowering) {
    ses::Field1D psi = ses::ho_eigenstate(kGrid, kOmega, 0);
    const ses::Field1D before = psi;
    double residual = 1.0;
    const double n2 = ses::ladder_fock(psi, kOmega, false, 8, &residual);
    EXPECT_LT(n2, 1e-9);
    EXPECT_LT(residual, 1e-10);
    for (int i = 0; i < psi.size(); ++i) {
        EXPECT_EQ(psi[i], before[i]);
    }
}

TEST(LadderFock, AgreesWithTheStableRungOnEigenstates) {
    ses::Field1D fock = ses::ho_eigenstate(kGrid, kOmega, 7);
    ses::Field1D rung = fock;
    double residual = 1.0;
    const double n2f = ses::ladder_fock(fock, kOmega, true, 16, &residual);
    const double n2r = ses::ladder_rung_stable(rung, kOmega, 7, true);
    EXPECT_NEAR(n2f, 8.0, 1e-9);
    EXPECT_NEAR(n2r, 8.0, 1e-6);
    EXPECT_NEAR(overlap_sq(fock, rung), 1.0, 1e-10);
}

TEST(Ladder, GroundStateSetupHasEnergyHalfOmega) {
    const ses::Field1D psi = ho_ground();
    const std::vector<double> v = ses::harmonic_potential(kGrid, kOmega);
    EXPECT_NEAR(ses::mean_energy(psi, v), 0.5 * kOmega, 1e-10);
}

TEST(Ladder, RaiseNormSqCountsNPlusOneAndEnergyClimbsTheFockChain) {
    ses::Field1D psi = ho_ground();
    const std::vector<double> v = ses::harmonic_potential(kGrid, kOmega);
    for (int n = 0; n < 8; ++n) {
        const double norm2 = ses::ladder_raise(psi, kOmega);
        EXPECT_NEAR(norm2, static_cast<double>(n + 1), 1e-9)
            << "raise from n = " << n;
        EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12) << "renormalized after raise";
        EXPECT_NEAR(ses::mean_energy(psi, v), (n + 1 + 0.5) * kOmega, 1e-9)
            << "energy after raise to n = " << n + 1;
    }
}

TEST(Ladder, LowerNormSqCountsNAndStepsBackDown) {
    ses::Field1D psi = ho_ground();
    for (int n = 0; n < 3; ++n) {
        ses::ladder_raise(psi, kOmega);
    }
    const std::vector<double> v = ses::harmonic_potential(kGrid, kOmega);
    for (int n = 3; n > 0; --n) {
        const double norm2 = ses::ladder_lower(psi, kOmega);
        EXPECT_NEAR(norm2, static_cast<double>(n), 1e-9) << "lower from n = " << n;
        EXPECT_NEAR(ses::mean_energy(psi, v), (n - 1 + 0.5) * kOmega, 1e-9)
            << "energy after lower to n = " << n - 1;
    }
}

TEST(Ladder, LowerOnGroundIsAnnihilationAndLeavesPsiUntouched) {
    ses::Field1D psi = ho_ground();
    const ses::Field1D before = psi;
    const double norm2 = ses::ladder_lower(psi, kOmega);
    EXPECT_LT(norm2, 1e-9) << "a|0> = 0 up to discretization";
    for (int i = 0; i < psi.size(); ++i) {
        EXPECT_EQ(psi[i], before[i]) << "annihilated apply must not write psi";
    }
}

TEST(Ladder, RaiseThenLowerRoundTripsToTheGroundState) {
    ses::Field1D psi = ho_ground();
    const ses::Field1D ground = psi;
    ses::ladder_raise(psi, kOmega);
    ses::ladder_lower(psi, kOmega);
    EXPECT_NEAR(overlap_sq(psi, ground), 1.0, 1e-10);
}

TEST(Ladder, ChainIsOmegaGeneric) {
    // stiffer well (omega = 0.75): noise gain k_max/sqrt(2 omega) is LOWER
    // than at 0.25, so the chain climbs just as cleanly.
    const double w = 0.75;
    ses::Field1D psi =
        ses::gaussian_wavepacket(kGrid, 0.0, 1.0 / std::sqrt(2.0 * w), 0.0);
    const std::vector<double> v = ses::harmonic_potential(kGrid, w);
    for (int n = 0; n < 4; ++n) {
        EXPECT_NEAR(ses::ladder_raise(psi, w), n + 1.0, 1e-9);
        EXPECT_NEAR(ses::mean_energy(psi, v), (n + 1.5) * w, 1e-9);
    }
}

// Independent measurement of the clean cap: raise from ground, stop at the
// first level where the ladder state diverges from the oracle (spectral noise)
// OR the oracle stops being grid-representable (energy off (n+1/2)w). Cap = min.
int measure_clean_cap(double w, const ses::Grid1D& g) {
    const std::vector<double> v = ses::harmonic_potential(g, w);
    ses::Field1D psi = ses::ho_eigenstate(g, w, 0);
    int cap = 0;
    for (int n = 1; n <= 60; ++n) {
        ses::ladder_raise(psi, w);
        const ses::Field1D oracle = ses::ho_eigenstate(g, w, n);
        const double e = ses::mean_energy(oracle, v);
        const double e_exact = (n + 0.5) * w;
        const bool oracle_representable =
            std::abs(e - e_exact) < 1e-3 * e_exact;
        if (!oracle_representable) {
            break;  // grid band ceiling
        }
        const double defect = 1.0 - overlap_sq(psi, oracle);
        if (defect > 1e-6) {  // ~0.1% amplitude corruption
            break;
        }
        cap = n;
    }
    return cap;
}

TEST(LadderCap, MatchesTheIndependentlyMeasuredCleanCap) {
    // Cross-check ladder_cap against measure_clean_cap -- an INDEPENDENT
    // reimplementation -- across the omega sweep.
    for (double w : {0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0}) {
        EXPECT_LE(std::abs(ses::ladder_cap(kGrid, w) - measure_clean_cap(w, kGrid)),
                  1)
            << "omega=" << w;
    }
}

TEST(LadderCap, IsNonMonotonicPeakingNearMatchedNyquist) {
    // Non-monotonic: derivative gain k_max/sqrt(2w) (worse at small w) vs the
    // x-term gain x_max*sqrt(w/2) (worse at large w) balance at w ~ 1 on this
    // grid, where the clean cap peaks.
    const auto cap = [](double w) { return ses::ladder_cap(kGrid, w); };
    EXPECT_GT(cap(1.0), cap(0.05));
    EXPECT_GT(cap(1.0), cap(8.0));
    EXPECT_GE(cap(1.0), 14);
    EXPECT_LE(cap(0.05), 3);
    EXPECT_LE(cap(8.0), 9);
}

TEST(LadderCap, ReproducesTheRecordedMeasuredCurve) {
    // Regression lock: MEASURED clean caps on the scene grid (-20..20, 256),
    // recorded so a grid/tol/FFT-path change that shifts the range trips this.
    const struct {
        double w;
        int cap;
    } pts[] = {{0.05, 1}, {0.1, 5},  {0.25, 12}, {0.5, 14},
               {1.0, 16}, {2.0, 15}, {4.0, 13},  {8.0, 7}};
    for (const auto& p : pts) {
        EXPECT_LE(std::abs(ses::ladder_cap(kGrid, p.w) - p.cap), 1)
            << "omega=" << p.w;
    }
}

// RED: the deep-level seed-underflow wall. psi_0 ~ exp(-w x^2/2) underflows
// double past w x^2/2 > ~745, so the plain recurrence seeds EXACT ZEROS past
// |x| ~ 38.6/sqrt(w), and a zero seed stays zero at every level -- high-n
// states lose their outer lobes (n=1200, w=1: wall at x~38.6 vs turning points
// +-49, ~44% of probability MISSING). Fix pinned below: a scaled per-point
// (mantissa, power-of-two exponent) chain -- power-of-two scaling is exact, so
// the plain output stays bitwise intact -- pushing the ceiling to box vs
// Nyquist, nothing else.
constexpr double kDeepOmega = 1.0;
// -60..60/4096: Nyquist allows n ~ 5700, box n_box = w x_max^2/2 ~ 1800, both
// far above the seed wall (~710).
const ses::Grid1D kDeepGrid{-60.0, 60.0, 4096};

TEST(HoEigenstateDeep, SurvivesTheSeedUnderflowWall) {
    const std::vector<double> v =
        ses::harmonic_potential(kDeepGrid, kDeepOmega);
    const ses::Field1D psi = ses::ho_eigenstate(kDeepGrid, kDeepOmega, 1200);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-10);
    EXPECT_EQ(node_count(psi), 1200);
    const double e_exact = 1200.5 * kDeepOmega;
    EXPECT_NEAR(ses::mean_energy(psi, v), e_exact, 1e-3 * e_exact);
}

TEST(HoLevelCapDeep, ReachesTheBoxCeilingNotTheSeedFloor) {
    // Cap must land at the box ceiling n_box = w x_max^2/2 = 1800 -- not the
    // seed wall (~710), not a fixed probe bound (400). Faithfulness fades over
    // the Airy transition (~200 levels) around n_box, so a slight overhang is
    // honest.
    const int cap = ses::ho_level_cap(kDeepGrid, kDeepOmega);
    EXPECT_GE(cap, 1200);
    EXPECT_LT(cap, 2200);
    const std::vector<double> v =
        ses::harmonic_potential(kDeepGrid, kDeepOmega);
    const ses::Field1D at_cap = ses::ho_eigenstate(kDeepGrid, kDeepOmega, cap);
    const double e_exact = (cap + 0.5) * kDeepOmega;
    EXPECT_NEAR(ses::mean_energy(at_cap, v), e_exact, 1e-3 * e_exact);
    // CONTAINED: the energy check is BLIND to box truncation -- a clipped
    // Hermite slice still satisfies k^2/2 + V = E at every sample, so grid
    // energy stays within 1e-5 of (n+1/2)w even with turning points OUTSIDE the
    // box. So the eigenstate must actually LIVE in the box: edge density
    // negligible vs bulk.
    double edge = 0.0;
    double bulk = 0.0;
    for (int i = 0; i < kDeepGrid.n; ++i) {
        bulk = std::max(bulk, std::norm(at_cap[i]));
    }
    edge = std::max(std::norm(at_cap[0]),
                    std::norm(at_cap[kDeepGrid.n - 1]));
    EXPECT_LT(edge, 1e-6 * bulk);
}

TEST(LadderFockDeep, RaisesADeepPairBeyondTheSeedWall) {
    // (|1198>+|1200>)/sqrt(2) --adag--> (sqrt(1199)|1199>+sqrt(1201)|1201>);
    // counting norm^2 = (1199+1201)/2 = 1200. Energy vs the EXACT spectrum --
    // a basis and input broken the same way cannot fake it.
    const ses::Field1D a = ses::ho_eigenstate(kDeepGrid, kDeepOmega, 1198);
    const ses::Field1D b = ses::ho_eigenstate(kDeepGrid, kDeepOmega, 1200);
    ses::Field1D psi{kDeepGrid};
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = a[i] + b[i];
    }
    ses::normalize(psi);
    double residual = 1.0;
    const double norm2 =
        ses::ladder_fock(psi, kDeepOmega, true, 1210, &residual);
    EXPECT_NEAR(norm2, 1200.0, 1e-3 * 1200.0);
    EXPECT_LT(residual, 1e-9);
    const std::vector<double> v =
        ses::harmonic_potential(kDeepGrid, kDeepOmega);
    const double e_exact =
        (1199.0 * 1199.5 + 1201.0 * 1201.5) / 2400.0 * kDeepOmega;
    EXPECT_NEAR(ses::mean_energy(psi, v), e_exact, 1e-3 * e_exact);
}

TEST(LadderFock, WideBandStaysExactForALowState) {
    // Band top FAR above the state's support must change nothing:
    // (|0>+|1>)/sqrt(2) raised in a 1500-level band -> (|1>+sqrt(2)|2>)/sqrt(3),
    // counting norm^2 = (1+2)/2 = 1.5.
    const ses::Field1D g0 = ses::ho_eigenstate(kDeepGrid, kDeepOmega, 0);
    const ses::Field1D g1 = ses::ho_eigenstate(kDeepGrid, kDeepOmega, 1);
    ses::Field1D psi{kDeepGrid};
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = g0[i] + g1[i];
    }
    ses::normalize(psi);
    double residual = 1.0;
    const double norm2 =
        ses::ladder_fock(psi, kDeepOmega, true, 1500, &residual);
    EXPECT_NEAR(norm2, 1.5, 1e-9);
    EXPECT_LT(residual, 1e-10);
    const ses::Field1D e1 = ses::ho_eigenstate(kDeepGrid, kDeepOmega, 1);
    const ses::Field1D e2 = ses::ho_eigenstate(kDeepGrid, kDeepOmega, 2);
    ses::Field1D want{kDeepGrid};
    const double s3 = 1.0 / std::sqrt(3.0);
    for (int i = 0; i < want.size(); ++i) {
        want[i] = s3 * (e1[i] + std::sqrt(2.0) * e2[i]);
    }
    EXPECT_NEAR(overlap_sq(psi, want), 1.0, 1e-9);
}

TEST(Ladder, LowerOnSuperpositionKeepsTheReachablePart) {
    // psi = (|0>+|1>)/sqrt(2): a psi = (1/sqrt(2))|0>, NOT annihilated
    // (norm^2 = 1/2), result is pure ground.
    ses::Field1D ground = ho_ground();
    ses::Field1D excited = ground;
    ses::ladder_raise(excited, kOmega);
    ses::Field1D psi = ground;
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = (ground[i] + excited[i]);
    }
    ses::normalize(psi);
    const double norm2 = ses::ladder_lower(psi, kOmega);
    EXPECT_NEAR(norm2, 0.5, 1e-9);
    EXPECT_NEAR(overlap_sq(psi, ground), 1.0, 1e-9);
}

}  // namespace
