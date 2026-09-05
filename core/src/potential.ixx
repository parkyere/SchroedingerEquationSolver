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

// Integral of 1/r over the unit cube centered at the origin (Waldvogel 1976
// closed form; Hummer 1996 tabulates 2.3800774).
inline constexpr double kCoulombCellAverage = 2.3800773639795527;
// Cells within this many spacings of a nucleus take the cube average of
// -Z/r; beyond it the point value is h^4-close (2e-4 relative at 3 cells).
// A cell within rounding of the shell may take either branch (CPU double vs
// GPU float): that is the 2e-4 jump, nothing more.
inline constexpr double kCoulombAverageRadius = 3.0;

namespace detail {

// Waldvogel antiderivative, d^3F/dxdydz = 1/r, principal-value atan (NOT
// atan2: the branch shift of pi does not cancel over the corners). A corner
// with a zero coordinate takes the limits: x*y*ln(z + r) -> 0, x^2 atan -> 0.
// ln(a + r) for a < 0 cancels; (s2 = the other two squares) a + r = s2/(r - a).
inline double cube_corner_term(double x, double y, double z) {
    const double r = std::sqrt(x * x + y * y + z * z);
    auto lg = [r](double a, double s2) {
        return std::log(a >= 0.0 ? a + r : s2 / (r - a));
    };
    double f = 0.0;
    if (x * y != 0.0) {
        f += x * y * lg(z, x * x + y * y);
    }
    if (y * z != 0.0) {
        f += y * z * lg(x, y * y + z * z);
    }
    if (z * x != 0.0) {
        f += z * x * lg(y, z * z + x * x);
    }
    if (x != 0.0) {
        f -= 0.5 * x * x * std::atan(y * z / (x * r));
    }
    if (y != 0.0) {
        f -= 0.5 * y * y * std::atan(z * x / (y * r));
    }
    if (z != 0.0) {
        f -= 0.5 * z * z * std::atan(x * y / (z * r));
    }
    return f;
}

}  // namespace detail

// (1/h^3) integral over the cube [-h/2, h/2]^3 of 1/|r - d|: the potential
// of a homogeneous cube, exact for d inside the cube (the nucleus cell) as
// well as outside; even in d. Signed corner sum, (-1)^(lower limits).
inline double coulomb_cube_average(Vec3d d, double h) {
    const double lo[3] = {-0.5 * h - d.x, -0.5 * h - d.y, -0.5 * h - d.z};
    const double hi[3] = {0.5 * h - d.x, 0.5 * h - d.y, 0.5 * h - d.z};
    double acc = 0.0;
    for (int m = 0; m < 8; ++m) {
        const bool ux = (m & 1) != 0;
        const bool uy = (m & 2) != 0;
        const bool uz = (m & 4) != 0;
        const int lower = (ux ? 0 : 1) + (uy ? 0 : 1) + (uz ? 0 : 1);
        const double f = detail::cube_corner_term(ux ? hi[0] : lo[0],
                                                  uy ? hi[1] : lo[1],
                                                  uz ? hi[2] : lo[2]);
        acc += (lower % 2 == 0) ? f : -f;
    }
    return acc / (h * h * h);
}

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

// Finite-volume bare Coulomb superposed over centers: cells within
// kCoulombAverageRadius of a nucleus take the cube average of -Z/|r-c| (the
// nucleus cell included, wherever the nucleus sits in it), the rest the point
// value -Z/r (docs/ARCHITECTURE.md: why not soft-Coulomb). Translation
// invariant to the h^4 switch: moving nuclei (rotor) see no grid egg-box.
// Cubic cells (h = x spacing). An exact hit keeps the tabulated -Z*C/h.
inline std::vector<double> regularized_coulomb_potential(
    const Grid3D& g, double Z, const std::vector<Vec3d>& centers) {
    const double h = g.x.spacing();
    const double r_avg = kCoulombAverageRadius * h;
    const double center_v = -Z * kCoulombCellAverage / h;
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    for (const Vec3d& c : centers) {
        for_each_cell(g, [&](int i, int j, int k, int flat) {
            const Vec3d d{g.x.coord(i) - c.x, g.y.coord(j) - c.y,
                          g.z.coord(k) - c.z};
            const double r = length(d);
            double vc = -Z / r;
            if (r == 0.0) {
                vc = center_v;
            } else if (r < r_avg) {
                vc = -Z * coulomb_cube_average(d, h);
            }
            v[static_cast<std::size_t>(flat)] += vc;
        });
    }
    return v;
}

inline std::vector<double> regularized_coulomb_potential(const Grid3D& g, double Z, Vec3d c) {
    return regularized_coulomb_potential(g, Z, std::vector<Vec3d>{c});
}

// Nearest grid point per axis, clamped to valid coords (xmax is off the
// periodic grid). Static molecular centers snap here so the atlas/oracle
// energies stay at the tabulated on-point values.
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
