module;
#include <algorithm>
#include <cmath>
#include <type_traits>
export module ses.grid;


// Periodic grid (required by the split-operator Fourier propagator): xmax aliases
// to xmin -- not a grid point -- so spacing divides by n, not n-1.


export namespace ses {

struct Grid1D {
    double xmin{};
    double xmax{};
    int n{};

    constexpr int size() const noexcept { return n; }
    constexpr double spacing() const noexcept { return (xmax - xmin) / n; }
    constexpr double coord(int i) const noexcept { return xmin + i * spacing(); }
};

// Nearest lattice index to x (uniform axis: arithmetic, no scan); clamped.
inline int nearest_index(const Grid1D& g, double x) {
    const int i = static_cast<int>(std::round((x - g.xmin) / g.spacing()));
    return std::clamp(i, 0, g.n - 1);
}

// Flat layout x-fastest: x-lines contiguous for the 3D FFT, and matches the
// row-major order GPU 3D-texture uploads expect.
struct Grid3D {
    Grid1D x{};
    Grid1D y{};
    Grid1D z{};

    constexpr int size() const noexcept { return x.n * y.n * z.n; }
    constexpr int flat(int i, int j, int k) const noexcept { return i + x.n * (j + y.n * k); }
    constexpr double cell_volume() const noexcept { return x.spacing() * y.spacing() * z.spacing(); }
};

// Full-grid scan, z-outer x-inner (flat ascending; order fixed so reductions
// stay bitwise identical). fn(i, j, k, flat) or fn(i, j, k).
template <class Fn>
void for_each_cell(const Grid3D& g, Fn&& fn) {
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                if constexpr (std::is_invocable_v<Fn&, int, int, int, int>) {
                    fn(i, j, k, g.flat(i, j, k));
                } else {
                    fn(i, j, k);
                }
            }
        }
    }
}

}  // namespace ses
