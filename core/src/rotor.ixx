module;
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>
export module ses.rotor;
export import ses.vec;
export import ses.field;
export import ses.grid;
import ses.emission;
import ses.potential;


// Ehrenfest rigid LINEAR rotor: the H2+ nuclei as a classical unit axis n
// with angular momentum L PERPENDICULAR to n (a linear molecule carries no
// nuclear angular momentum along its axis -- that is the electronic Lambda),
// moment of inertia I = mu R^2. The electron cloud drives it through the
// orientation torque tau = -dE/d(orientation). CONTRACT: tests/rotor_test.cpp.


export namespace ses {

struct RigidRotor {
    Vec3d n{0.0, 0.0, 1.0};  // unit axis
    Vec3d L{};               // au (hbar = 1), L . n = 0
    double inertia = 1.0;    // mu R^2 (au)
};

namespace rotor_detail {

inline Vec3d perp_to(Vec3d v, Vec3d n) noexcept { return v - dot(v, n) * n; }

// Rodrigues: v rotated about the unit axis k by th, right-handed.
inline Vec3d rotate_about(Vec3d v, Vec3d k, double th) noexcept {
    const double c = std::cos(th);
    const double s = std::sin(th);
    return c * v + s * cross(k, v) + ((1.0 - c) * dot(k, v)) * k;
}

inline constexpr int kJScanMax = 100000;  // centrifugal-cap scan guard

}  // namespace rotor_detail

inline Vec3d rotor_omega(const RigidRotor& r) noexcept {
    return (1.0 / r.inertia) * r.L;
}

inline double rotor_energy(const RigidRotor& r) noexcept {
    return dot(r.L, r.L) / (2.0 * r.inertia);
}

// 2 pi I / |L|; +inf at rest.
inline double rotor_period(const RigidRotor& r) noexcept {
    const double l = length(r.L);
    return l > 0.0 ? 2.0 * std::numbers::pi * r.inertia / l
                   : std::numeric_limits<double>::infinity();
}

// L += dJ axis, projected perpendicular to n. Refused (false, L untouched)
// when |L| would exceed l_max AND grow -- slowing down is always allowed.
inline bool rotor_kick(RigidRotor& r, Vec3d axis, double dJ,
                       double l_max = std::numeric_limits<double>::infinity()) {
    const Vec3d next = rotor_detail::perp_to(r.L + dJ * axis, r.n);
    const double after = length(next);
    if (after > l_max && after > length(r.L)) {
        return false;
    }
    r.L = next;
    return true;
}

// Velocity-Verlet on the sphere: half-kick, EXACT rotation of n about omega,
// half-kick; L re-projected perpendicular to the new axis. Free rotation is
// exact for any dt (a torque enters only through the half-kicks).
inline void rotor_step(RigidRotor& r, Vec3d tau, double dt) {
    r.L = r.L + (0.5 * dt) * rotor_detail::perp_to(tau, r.n);
    const Vec3d w = rotor_omega(r);
    const double wn = length(w);
    if (wn > 0.0) {
        const Vec3d turned =
            rotor_detail::rotate_about(r.n, (1.0 / wn) * w, wn * dt);
        r.n = (1.0 / length(turned)) * turned;  // re-unit against round-off
    }
    r.L = rotor_detail::perp_to(
        r.L + (0.5 * dt) * rotor_detail::perp_to(tau, r.n), r.n);
}

// tau = sum_k r_k x F_k with r_k = +-(R/2) n and F_k the electron's pull on
// nucleus k: tau = (R/2) n x (F_1 - F_2) = -dE/d(orientation). Shared by the
// CPU path below and the GPU path (Engine::two_center_forces).
inline Vec3d rotor_torque_from_forces(double R, Vec3d n, Vec3d f1,
                                      Vec3d f2) noexcept {
    return (0.5 * R) * cross(n, f1 - f2);
}

// F_k = <psi| grad V_k |psi>, V_k = nucleus k's regularized unit-charge
// Coulomb term (central differences of the sampled V_k). psi normalized.
inline Vec3d rotor_torque(const Field3D& psi, double R, Vec3d n) {
    const Grid3D& g = psi.grid();
    const std::vector<double> v1 =
        regularized_coulomb_potential(g, 1.0, (0.5 * R) * n);
    const std::vector<double> v2 =
        regularized_coulomb_potential(g, 1.0, (-0.5 * R) * n);
    return rotor_torque_from_forces(R, n, mean_potential_gradient(psi, v1, g),
                                    mean_potential_gradient(psi, v2, g));
}

// Largest J whose V(R) + J(J+1)/(2 mu R^2) keeps a BOUND well: an interior
// local minimum below the dissociation asymptote (the bare curve's last
// sample). Wells held only by the centrifugal barrier are quasi-bound (they
// tunnel away) and do not count. -1 when even J = 0 is unbound.
inline int rotor_j_max(const std::vector<double>& r,
                       const std::vector<double>& v, double mu) {
    const auto bound = [&](int j) {
        const double c = j * (j + 1.0) / (2.0 * mu);
        const double asymptote = v.back();
        for (std::size_t i = 1; i + 1 < r.size(); ++i) {
            const double a = v[i - 1] + c / (r[i - 1] * r[i - 1]);
            const double b = v[i] + c / (r[i] * r[i]);
            const double d = v[i + 1] + c / (r[i + 1] * r[i + 1]);
            if (b < a && b < d && b < asymptote) {
                return true;
            }
        }
        return false;
    };
    if (r.size() < 3 || !bound(0)) {
        return -1;
    }
    for (int j = 0; j < rotor_detail::kJScanMax; ++j) {
        if (!bound(j + 1)) {
            return j;
        }
    }
    return rotor_detail::kJScanMax;
}

}  // namespace ses
