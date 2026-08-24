module;
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <vector>
export module ses.spectral;
import ses.grid;
import ses.parallel;


// FFT-bin -> physical wavenumber (fftfreq layout) + shared Strang split-step
// table builders and elementwise phase apply (propagator/imaginary_time/bloch/drive).


export namespace ses {

inline std::vector<double> wavenumbers(const Grid1D& g) {
    const double dk = 2.0 * std::numbers::pi / (g.xmax - g.xmin);
    std::vector<double> k(static_cast<std::size_t>(g.n));
    for (int j = 0; j < g.n; ++j) {
        // 2j<n not j<n/2: keeps DC=0 at n=1, matches fftfreq split for all n.
        const int shifted = (2 * j < g.n) ? j : j - g.n;
        k[static_cast<std::size_t>(j)] = dk * shifted;
    }
    return k;
}

// e^{i theta}
inline std::complex<double> unit_phase(double theta) noexcept {
    return std::complex<double>{std::cos(theta), std::sin(theta)};
}

// table[i] = elem(-0.5 v[i] scale); elem = unit_phase (dt) or exp (dtau).
template <class Elem>
auto build_half_potential_table(const std::vector<double>& v, double scale,
                                Elem&& elem) {
    std::vector<decltype(elem(0.0))> table(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        table[i] = elem(-0.5 * v[i] * scale);
    }
    return table;
}

// table[j] = elem(-0.5 k^2 / mass * scale), fftfreq order.
template <class Elem>
auto build_kinetic_table(const Grid1D& g, double mass, double scale, Elem&& elem) {
    const std::vector<double> k = wavenumbers(g);
    std::vector<decltype(elem(0.0))> table(k.size());
    for (std::size_t j = 0; j < k.size(); ++j) {
        table[j] = elem(-0.5 * (k[j] * k[j]) / mass * scale);
    }
    return table;
}

template <class Elem>
auto build_kinetic_table(const Grid3D& g, double mass, double scale, Elem&& elem) {
    const std::vector<double> kx = wavenumbers(g.x);
    const std::vector<double> ky = wavenumbers(g.y);
    const std::vector<double> kz = wavenumbers(g.z);
    std::vector<decltype(elem(0.0))> table(static_cast<std::size_t>(g.size()));
    for_each_cell(g, [&](int i, int j, int k, int flat) {
        const double kxx = kx[static_cast<std::size_t>(i)];
        const double kyy = ky[static_cast<std::size_t>(j)];
        const double kzz = kz[static_cast<std::size_t>(k)];
        table[static_cast<std::size_t>(flat)] =
            elem(-0.5 * (kxx * kxx + kyy * kyy + kzz * kzz) / mass * scale);
    });
    return table;
}

// Elementwise (disjoint) multiply, threaded: bitwise identical to serial.
inline void apply_phase(const std::vector<std::complex<double>>& phase,
                        std::vector<std::complex<double>>& a) noexcept {
    parallel_for(static_cast<int>(a.size()), [&](int i) {
        a[static_cast<std::size_t>(i)] =
            a[static_cast<std::size_t>(i)] * phase[static_cast<std::size_t>(i)];
    });
}

}  // namespace ses
