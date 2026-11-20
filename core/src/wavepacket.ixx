module;
#include <complex>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>
export module ses.wavepacket;
export import ses.field;
export import ses.grid;
export import ses.vec;


// Analytic prefactor is unit-norm; normalize() absorbs discrete sampling error.


export namespace ses {

inline Field1D gaussian_wavepacket(const Grid1D& g, double x0, double sigma, double k0) {
    Field1D psi{g};
    const double amp = std::pow(2.0 * std::numbers::pi * sigma * sigma, -0.25);
    for (int i = 0; i < psi.size(); ++i) {
        const double x = g.coord(i);
        const double envelope = amp * std::exp(-(x - x0) * (x - x0) / (4.0 * sigma * sigma));
        psi[i] = std::complex<double>{envelope * std::cos(k0 * x), envelope * std::sin(k0 * x)};
    }
    normalize(psi);
    return psi;
}

inline Field3D gaussian_wavepacket(const Grid3D& g, Vec3d r0, Vec3d sigma, Vec3d k0) {
    Field3D psi{g};
    auto envelope = [](double u, double u0, double s) {
        return std::pow(2.0 * std::numbers::pi * s * s, -0.25) *
               std::exp(-(u - u0) * (u - u0) / (4.0 * s * s));
    };
    // The envelope is a product of per-axis factors: tabulate them (3n pow/exp
    // instead of 3n^3) and keep the product order, so every cell is bitwise
    // what the per-cell evaluation gave; the rows then fill on the pool.
    const auto axis_envelope = [&envelope](const Grid1D& a, double u0, double s) {
        std::vector<double> t(static_cast<std::size_t>(a.n));
        for (int p = 0; p < a.n; ++p) {
            t[static_cast<std::size_t>(p)] = envelope(a.coord(p), u0, s);
        }
        return t;
    };
    const std::vector<double> ex = axis_envelope(g.x, r0.x, sigma.x);
    const std::vector<double> ey = axis_envelope(g.y, r0.y, sigma.y);
    const std::vector<double> ez = axis_envelope(g.z, r0.z, sigma.z);
    parallel_for_each_cell(g, [&](int i, int j, int k) {
        const double x = g.x.coord(i);
        const double y = g.y.coord(j);
        const double z = g.z.coord(k);
        const double env = ex[static_cast<std::size_t>(i)] *
                           ey[static_cast<std::size_t>(j)] *
                           ez[static_cast<std::size_t>(k)];
        const double phase = k0.x * x + k0.y * y + k0.z * z;
        psi(i, j, k) = std::complex<double>{env * std::cos(phase), env * std::sin(phase)};
    });
    normalize(psi);
    return psi;
}

}  // namespace ses
