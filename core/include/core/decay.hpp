#pragma once

// Spontaneous decay via quantum jumps (transitions arc T4, the Monte-Carlo-
// wavefunction picture). The Schrodinger equation carries no lifetimes; the
// decay RATE follows from the computed spectrum through the Einstein A
// coefficient (atomic units):
//     A = (4/3) alpha^3 omega^3 |<f|r|i>|^2
// so selection rules enter automatically (forbidden channel -> strength 0 ->
// A = 0, e.g. the metastable 2s). Jumps are a Poisson process weighted by
// the CURRENT excited population; randomness is injected by the caller.

#include <core/complex.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>

#include <cmath>
#include <cstddef>

namespace ses {

inline constexpr double kFineStructureConstant = 7.2973525693e-3;

struct DipoleMatrixElement {
    Complex<double> x;
    Complex<double> y;
    Complex<double> z;
};

// <f| r |i> component-wise: sum conj(f) * r * i * dV.
inline DipoleMatrixElement dipole_matrix_element(const Field3D& f, const Field3D& i) {
    const Grid3D& g = f.grid();
    Complex<double> dx{};
    Complex<double> dy{};
    Complex<double> dz{};
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int ii = 0; ii < g.x.n; ++ii) {
                const Complex<double> t = std::conj(f(ii, j, k)) * i(ii, j, k);
                dx += g.x.coord(ii) * t;
                dy += g.y.coord(j) * t;
                dz += g.z.coord(k) * t;
            }
        }
    }
    const double dv = g.cell_volume();
    return DipoleMatrixElement{dv * dx, dv * dy, dv * dz};
}

inline double dipole_strength_sq(const DipoleMatrixElement& d) {
    return std::norm(d.x) + std::norm(d.y) + std::norm(d.z);
}

// Einstein A coefficient (atomic units): the spontaneous decay rate.
inline double einstein_a(double omega, double dipole_sq) {
    const double a3 = kFineStructureConstant * kFineStructureConstant *
                      kFineStructureConstant;
    return (4.0 / 3.0) * a3 * omega * omega * omega * dipole_sq;
}

struct JumpResult {
    bool jumped{};
    double p_jump{};
};

// One Poisson decay trial over an interval dt: jump probability
// p = 1 - exp(-gamma * P_e * dt) with P_e = |<excited|psi>|^2. On a jump the
// state collapses onto the ground state (the photon carries the rest away);
// on survival psi is untouched. u in [0,1) is the caller's uniform draw.
inline JumpResult quantum_jump(Field3D& psi, const Field3D& excited, const Field3D& ground,
                               double gamma, double dt, double u) {
    const double p_e = norm_sq(inner_product(excited, psi));
    const double p = 1.0 - std::exp(-gamma * p_e * dt);
    if (u < p) {
        psi = ground;
        return JumpResult{true, p};
    }
    return JumpResult{false, p};
}

}  // namespace ses
