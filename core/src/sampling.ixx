module;
#include <complex>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
export module ses.sampling;
export import ses.grid;
export import ses.vec;
export import ses.marching_cubes;
export import ses.field;
export import ses.colormap;


// atan2 the interpolated COMPLEX value, not the phase directly: constant-phase
// regions stay constant, amplitude cancels in the ratio.


export namespace ses {

namespace sampling_detail {

struct Cell {
    int i;     // lower cell
    int next;  // upper cell (== i on a collapsed axis: no neighbour to read)
    double t;  // lerp weight; t=1 covers the last cell
};

// clamp i to n-2 so next stays in range (n = 1: the single cell, weight 0).
inline Cell cell_and_t(double u, const Grid1D& axis) noexcept {
    if (axis.n < 2) {
        return {0, 0, 0.0};
    }
    const double s = (u - axis.xmin) / axis.spacing();
    int i = static_cast<int>(std::floor(s));
    i = std::clamp(i, 0, axis.n - 2);
    return {i, i + 1, s - i};
}

}  // namespace sampling_detail

inline std::complex<double> sample_trilinear(const Field3D& f, Vec3d p) noexcept {
    const Grid3D& g = f.grid();
    const auto [i, i1, tx] = sampling_detail::cell_and_t(p.x, g.x);
    const auto [j, j1, ty] = sampling_detail::cell_and_t(p.y, g.y);
    const auto [k, k1, tz] = sampling_detail::cell_and_t(p.z, g.z);

    auto lerp = [](std::complex<double> a, std::complex<double> b, double t) {
        return a + t * (b - a);
    };

    const std::complex<double> c00 = lerp(f(i, j, k), f(i1, j, k), tx);
    const std::complex<double> c10 = lerp(f(i, j1, k), f(i1, j1, k), tx);
    const std::complex<double> c01 = lerp(f(i, j, k1), f(i1, j, k1), tx);
    const std::complex<double> c11 = lerp(f(i, j1, k1), f(i1, j1, k1), tx);
    return lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz);
}

inline std::vector<Rgb> phase_colors(const Mesh& mesh, const Field3D& psi) {
    std::vector<Rgb> colors;
    colors.reserve(mesh.vertices.size());
    for (const Vec3d& v : mesh.vertices) {
        const std::complex<double> s = sample_trilinear(psi, v);
        colors.push_back(phase_color(std::atan2(s.imag(), s.real())));
    }
    return colors;
}

}  // namespace ses
