module;
#include <numbers>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
export module ses.potential;
export import ses.grid;
export import ses.vec;
import ses.parallel;


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

// Si(x) = integral_0^x sin t / t dt: term-recursive series below 4 (largest
// term ~8, fine in float too), Abramowitz-Stegun 5.2.38/39 rational f, g
// above (composite |err| 5.7e-7 double, 7.5e-7 float32, worst near x ~ 13).
// Same formula in two_center.slang. CONTRACT: potential_test SineIntegral.
inline double sine_integral(double x) {
    if (x < 0.0) {
        return -sine_integral(-x);
    }
    if (x <= 4.0) {
        const double x2 = x * x;
        double term = x;  // x^(2n+1) / (2n+1)!
        double s = 0.0;
        for (int n = 0; n < 24; ++n) {
            s += term / (2 * n + 1);
            term *= -x2 / ((2 * n + 2) * (2 * n + 3));
        }
        return s;
    }
    const double x2 = x * x;
    const double fn = (((x2 + 38.027264) * x2 + 265.187033) * x2 + 335.677320) * x2 + 38.102495;
    const double fd = (((x2 + 40.021433) * x2 + 322.624911) * x2 + 570.236280) * x2 + 157.105423;
    const double gn = (((x2 + 42.242855) * x2 + 302.757865) * x2 + 352.018498) * x2 + 21.821899;
    const double gd = (((x2 + 48.196927) * x2 + 482.485984) * x2 + 1114.978885) * x2 + 449.690326;
    const double f = fn / fd / x;
    const double g = gn / gd / x2;
    return 0.5 * std::numbers::pi - f * std::cos(x) - g * std::sin(x);
}

// Unit-charge Coulomb band-limited to the grid's Nyquist sphere K = pi/h:
// (2/pi) Si(K r) / r, -> 2/h at r = 0. Its Gibbs tail ~cos(K r)/(K r^2) is
// exactly the content a spacing-h lattice aliases, so the lattice sum is
// translation invariant (egg-box 0.27 mHa at h = 0.31 vs 3.8 for the cube
// average, 15.8 point-sampled) and the relaxed 1s lands +1.8 mHa above -0.5
// at h = 0.31 (box-converged; the +-5 test box reads -0.4996, its periodic
// images worth -1.5 mHa). Not softening: no free parameter, depth
// grid-dictated.
inline double band_limited_coulomb(double r, double h) {
    const double k = std::numbers::pi / h;
    return r > 0.0 ? 2.0 / std::numbers::pi * sine_integral(k * r) / r
                   : 2.0 / h;
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

// Band-limited bare Coulomb superposed over centers: every cell takes
// -Z (2/pi) Si(pi r/h) / r for every nucleus (r = 0: -2Z/h), no window
// (docs/ARCHITECTURE.md: why not soft-Coulomb). Cubic cells (h = x spacing).
// Verbatim on the GPU (two_center.slang). One Si per (cell, center): z-slabs
// in parallel (each cell written by one worker -> bitwise deterministic).
inline std::vector<double> regularized_coulomb_potential(
    const Grid3D& g, double Z, const std::vector<Vec3d>& centers) {
    const double h = g.x.spacing();
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    parallel_for(g.z.n, [&](int k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                double acc = 0.0;
                for (const Vec3d& c : centers) {
                    const Vec3d d{g.x.coord(i) - c.x, g.y.coord(j) - c.y,
                                  g.z.coord(k) - c.z};
                    acc -= Z * band_limited_coulomb(length(d), h);
                }
                v[static_cast<std::size_t>(g.flat(i, j, k))] = acc;
            }
        }
    });
    return v;
}

inline std::vector<double> regularized_coulomb_potential(const Grid3D& g, double Z, Vec3d c) {
    return regularized_coulomb_potential(g, Z, std::vector<Vec3d>{c});
}

// Cap |V| so every split-operator kick e^{-iV dt/2} stays below max_phase
// rad: a kick past pi ALIASES into a potential of the opposite sign (the
// Rutherford +2Z/h core at dt = 0.01 read 5.06 rad = a -245 Ha trap). Only
// for cells the physics never reaches; reachable cells must stay under the
// budget on their own. CONTRACT: potential_test ClampTrotterPhase.
inline std::vector<double> clamp_trotter_phase(std::vector<double> v, double dt,
                                               double max_phase) {
    const double cap = 2.0 * max_phase / dt;
    for (double& x : v) {
        x = std::clamp(x, -cap, cap);
    }
    return v;
}

// Nearest grid point per axis, clamped to valid coords (xmax is off the
// periodic grid). Static molecular centers snap here so their on-point
// energies are reproducible grid to grid.
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
