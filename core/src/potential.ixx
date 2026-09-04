module;
#include <numbers>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
export module ses.potential;
export import ses.grid;
export import ses.vec;


// Potential builders. Atomic units.


export namespace ses {

inline std::vector<double> harmonic_potential(const Grid1D& g, double omega, double x0 = 0.0) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double dx = g.coord(i) - x0;
        v[static_cast<std::size_t>(i)] = 0.5 * omega * omega * dx * dx;
    }
    return v;
}

inline std::vector<double> soft_coulomb_potential(const Grid1D& g, double Z, double a,
                                                  double x0 = 0.0) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double dx = g.coord(i) - x0;
        v[static_cast<std::size_t>(i)] = -Z / std::sqrt(dx * dx + a * a);
    }
    return v;
}

inline std::vector<double> harmonic_potential(const Grid3D& g, double omega, Vec3d c) {
    std::vector<double> v(static_cast<std::size_t>(g.size()));
    for_each_cell(g, [&](int i, int j, int k, int flat) {
        const double dx = g.x.coord(i) - c.x;
        const double dy = g.y.coord(j) - c.y;
        const double dz = g.z.coord(k) - c.z;
        v[static_cast<std::size_t>(flat)] =
            0.5 * omega * omega * (dx * dx + dy * dy + dz * dz);
    });
    return v;
}

inline std::vector<double> soft_coulomb_potential(const Grid3D& g, double Z, double a, Vec3d c) {
    std::vector<double> v(static_cast<std::size_t>(g.size()));
    for_each_cell(g, [&](int i, int j, int k, int flat) {
        const double dx = g.x.coord(i) - c.x;
        const double dy = g.y.coord(j) - c.y;
        const double dz = g.z.coord(k) - c.z;
        v[static_cast<std::size_t>(flat)] =
            -Z / std::sqrt(dx * dx + dy * dy + dz * dz + a * a);
    });
    return v;
}

inline std::vector<double> double_well_potential(const Grid1D& g, double vb, double a) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double s = g.coord(i) / a;
        const double q = s * s - 1.0;
        v[static_cast<std::size_t>(i)] = vb * q * q;
    }
    return v;
}

// reflectionless at v0 = l(l+1)/(2 a^2).
inline std::vector<double> poschl_teller_potential(const Grid1D& g, double v0,
                                                   double a, double x0 = 0.0) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double c = std::cosh((g.coord(i) - x0) / a);
        v[static_cast<std::size_t>(i)] = -v0 / (c * c);
    }
    return v;
}

// E_n = w0 (n + 1/2) - (alpha^2/2)(n + 1/2)^2, w0 = alpha sqrt(2 d).
inline std::vector<double> morse_potential(const Grid1D& g, double d,
                                           double alpha, double x0) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double q = 1.0 - std::exp(-alpha * (g.coord(i) - x0));
        v[static_cast<std::size_t>(i)] = d * q * q;
    }
    return v;
}

// Integral of 1/r over the unit cube centered at origin; 7-digit quadrature.
// Uniform static E folded into the potential: V += e0 * coord_axis (odd in
// e0; axis 0=x 1=y 2=z). Relax tables MUST ride this same effective V or
// imaginary time cools to the field-free ground.
inline std::vector<double> tilted_potential(std::vector<double> v,
                                            const Grid3D& g, double e0,
                                            int axis) {
    const Grid1D* axes[3] = {&g.x, &g.y, &g.z};
    const Grid1D& ax = *axes[axis];
    std::vector<double> tilt(static_cast<std::size_t>(ax.n));
    for (int p = 0; p < ax.n; ++p) {
        tilt[static_cast<std::size_t>(p)] = e0 * ax.coord(p);
    }
    for_each_cell(g, [&](int i, int j, int k, int flat) {
        const int idx[3] = {i, j, k};
        v[static_cast<std::size_t>(flat)] +=
            tilt[static_cast<std::size_t>(idx[axis])];
    });
    return v;
}

inline constexpr double kCoulombCellAverage = 2.3800774;

inline std::vector<double> barrier_potential(const Grid1D& g, double v0,
                                             double x_lo, double x_hi) {
    std::vector<double> v(static_cast<std::size_t>(g.n), 0.0);
    for (int i = 0; i < g.n; ++i) {
        const double x = g.coord(i);
        if (x >= x_lo && x < x_hi) {
            v[static_cast<std::size_t>(i)] = v0;
        }
    }
    return v;
}

inline std::vector<double> barrier_potential(const Grid3D& g, double v0,
                                             double x_lo, double x_hi) {
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    for (int i = 0; i < g.x.n; ++i) {
        const double x = g.x.coord(i);
        if (x < x_lo || x >= x_hi) {
            continue;
        }
        for (int k = 0; k < g.z.n; ++k) {
            for (int j = 0; j < g.y.n; ++j) {
                v[static_cast<std::size_t>(g.flat(i, j, k))] = v0;
            }
        }
    }
    return v;
}

// Bare -Z/|r-c| superposed over centers: each exact-hit nucleus cell takes the
// analytic cell average, others add plain -Z/r (docs/ARCHITECTURE.md: why not
// soft-Coulomb). Requires cubic cells + centers on grid points (BO molecular
// potentials, e.g. H2+); an off-point nucleus just gets -Z/r throughout.
inline std::vector<double> regularized_coulomb_potential(
    const Grid3D& g, double Z, const std::vector<Vec3d>& centers) {
    const double h = g.x.spacing();
    const double center_v = -Z * kCoulombCellAverage / h;
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    for (const Vec3d& c : centers) {
        for_each_cell(g, [&](int i, int j, int k, int flat) {
            const double dx = g.x.coord(i) - c.x;
            const double dy = g.y.coord(j) - c.y;
            const double dz = g.z.coord(k) - c.z;
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            // Cell-average cap for the whole cell holding the nucleus (r < h/2):
            // a moving nucleus must never expose -Z/r -> -inf to the Trotter
            // step; on-grid nuclei keep every value (neighbours sit h away).
            v[static_cast<std::size_t>(flat)] += (r < 0.5 * h) ? center_v : -Z / r;
        });
    }
    return v;
}

inline std::vector<double> regularized_coulomb_potential(const Grid3D& g, double Z, Vec3d c) {
    return regularized_coulomb_potential(g, Z, std::vector<Vec3d>{c});
}

// Nearest grid point per axis, clamped to valid coords (xmax is off the
// periodic grid). Molecular centers MUST snap here: the multi-center builder
// regularizes only exact-hit cells, so an off-grid center gets an arbitrary
// -Z/r depth.
inline Vec3d snap_to_grid(const Grid3D& g, Vec3d p) {
    auto axis = [](const Grid1D& ax, double x) {
        const double h = ax.spacing();
        double i = std::round((x - ax.xmin) / h);
        i = std::min(std::max(i, 0.0), static_cast<double>(ax.n - 1));
        return ax.xmin + i * h;
    };
    return {axis(g.x, p.x), axis(g.y, p.y), axis(g.z, p.z)};
}

// 1 in the interior, ramps to 0 within `width` of each wall. Multiply psi
// each real-time step to damp outgoing flux; NEVER during imaginary-time prep.
inline std::vector<double> absorbing_mask(const Grid1D& g, double width) {
    std::vector<double> m(static_cast<std::size_t>(g.n));
    if (g.n == 1) {
        m[0] = 1.0;  // collapsed axis: no walls
        return m;
    }
    for (int i = 0; i < g.n; ++i) {
        const double x = g.coord(i);
        const double d_lo = x - g.xmin;
        const double d_hi = g.xmax - x;
        const double d = d_lo < d_hi ? d_lo : d_hi;
        if (d >= width) {
            m[static_cast<std::size_t>(i)] = 1.0;
            continue;
        }
        const double t = d / width;
        const double s = std::sin(0.5 * std::numbers::pi * t);
        m[static_cast<std::size_t>(i)] = s * s;
    }
    return m;
}

// Quadratic CAP frame on the x/y box edges (k = 0 plane; 2D scenes, nz = 1):
// w = w0 (1 - d/width)^2 inside the ramp, mask = exp(-w dt) per step.
// Quadratic, not cos^2: cos^2 is too stiff for slow packets (~30% reflection
// at k0 = 1).
inline std::vector<double> quadratic_cap_mask(const Grid3D& g, double width,
                                              double w0, double dt) {
    auto ramp_w = [&](const Grid1D& ax, double x) {
        const double d = std::min(x - ax.xmin, ax.xmax - x);
        if (d >= width) {
            return 0.0;
        }
        const double t = 1.0 - d / width;
        return w0 * t * t;
    };
    std::vector<double> m(static_cast<std::size_t>(g.size()));
    for (int j = 0; j < g.y.n; ++j) {
        const double wy = ramp_w(g.y, g.y.coord(j));
        for (int i = 0; i < g.x.n; ++i) {
            const double wx = ramp_w(g.x, g.x.coord(i));
            m[static_cast<std::size_t>(g.flat(i, j, 0))] =
                std::exp(-(wx + wy) * dt);
        }
    }
    return m;
}

// Separable: product of per-axis 1D ramps -- 3n sin calls, not 3n^3.
inline std::vector<double> absorbing_mask(const Grid3D& g, double width) {
    const std::vector<double> mx = absorbing_mask(g.x, width);
    const std::vector<double> my = absorbing_mask(g.y, width);
    const std::vector<double> mz = absorbing_mask(g.z, width);
    std::vector<double> m(static_cast<std::size_t>(g.size()));
    for_each_cell(g, [&](int i, int j, int k, int flat) {
        m[static_cast<std::size_t>(flat)] = mx[static_cast<std::size_t>(i)] *
                                            my[static_cast<std::size_t>(j)] *
                                            mz[static_cast<std::size_t>(k)];
    });
    return m;
}

}  // namespace ses
