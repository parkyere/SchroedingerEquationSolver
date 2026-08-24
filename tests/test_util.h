// Shared test scaffolding. Plain header: include AFTER the ses module imports
// (the gated helpers name ses:: types those modules export). Carries NO std
// includes -- textual std headers after an import miscompile on MSVC, so the
// consumer TU must include <algorithm> <cmath> <complex> <cstddef> <vector>
// BEFORE its imports (the repo's include-before-import rule).
// Opt-in sections, chosen by defining before inclusion:
//   SES_TEST_UTIL_HARMONIC  cube + harmonic_state (needs ses.grid + ses.field)
//   SES_TEST_UTIL_RADIAL    bare -1/r radial recipe (needs ses.radial)
#ifndef SES_TESTS_TEST_UTIL_H
#define SES_TESTS_TEST_UTIL_H

namespace ses_test {

// Largest |a_i - b_i| over the raw complex samples (spinexact max_amp_diff
// pattern).
template <class FieldLike>
double max_abs_diff(const FieldLike& a, const FieldLike& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.data().size(); ++i) {
        m = std::max(m, std::abs(a.data()[i] - b.data()[i]));
    }
    return m;
}

#ifdef SES_TEST_UTIL_HARMONIC

inline ses::Grid3D cube(double lo, double hi, int n) {
    return ses::Grid3D{ses::Grid1D{lo, hi, n}, ses::Grid1D{lo, hi, n},
                       ses::Grid1D{lo, hi, n}};
}

// Lowest 3D harmonic-oscillator eigenstates; normalize absorbs the analytic
// prefactors. axis: -1 = ground, 0/1/2 = one-quantum excitation along x/y/z.
inline ses::Field3D harmonic_state(const ses::Grid3D& g, double w0, int axis) {
    ses::Field3D f{g};
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double y = g.y.coord(j);
                const double z = g.z.coord(k);
                const double env =
                    std::exp(-0.5 * w0 * (x * x + y * y + z * z));
                const double q = axis == 0   ? x
                                 : axis == 1 ? y
                                 : axis == 2 ? z
                                             : 1.0;
                f(i, j, k) = std::complex<double>{q * env, 0.0};
            }
        }
    }
    ses::normalize(f);
    return f;
}

#endif  // SES_TEST_UTIL_HARMONIC

#ifdef SES_TEST_UTIL_RADIAL

// Radial solve box shared with the app: mirrors ses_shell::kTrapBox = 20.0 and
// ses_shell::kTrapRadialSamples = 3999 (scenario/src/harmonic_director.ixx).
// Tests must not import scenario, so keep these in sync by hand.
inline constexpr double kRadialBoxRMax = 20.0;
inline constexpr int kRadialBoxSamples = 3999;

// Bare -1/r hydrogen radial problem, the app's collapse-target construction.
struct BareHydrogenRadial {
    ses::RadialGrid rg;
    std::vector<double> vr;  // -1/r sampled on rg (r = (i+1) h > 0)
};

inline BareHydrogenRadial bare_hydrogen_radial(
    double rmax = kRadialBoxRMax, int n = kRadialBoxSamples) {
    BareHydrogenRadial b{ses::RadialGrid{rmax, n},
                         std::vector<double>(static_cast<std::size_t>(n))};
    for (int i = 0; i < b.rg.n; ++i) {
        b.vr[static_cast<std::size_t>(i)] = -1.0 / b.rg.r(i);
    }
    return b;
}

#endif  // SES_TEST_UTIL_RADIAL

}  // namespace ses_test

#endif  // SES_TESTS_TEST_UTIL_H
