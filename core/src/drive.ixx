module;
#include <complex>
#include <cmath>
#include <cstddef>
#include <vector>
#include <cstdint>
export module ses.drive;
export import ses.grid;
export import ses.vec;
export import ses.propagator;
export import ses.fft;
export import ses.field;


// V_drive = amplitude*(axis.r)*cos(omega t); dipole half-kicks wrap prop's
// static Strang split-step, preserving O(dt^2).
export namespace ses {

struct DipoleDrive {
    Vec3d axis;          // polarization; need not be unit (scales E0)
    double amplitude{};  // E0, atomic units
    double omega{};
};

inline void apply_dipole_halfkick(Field3D& psi, const DipoleDrive& d, double t, double dt) noexcept {
    const double theta = d.amplitude * std::cos(d.omega * t) * 0.5 * dt;
    const Grid3D& g = psi.grid();
    for_each_cell(g, [&](int i, int j, int k) {
        const double u = d.axis.x * g.x.coord(i) + d.axis.y * g.y.coord(j) +
                         d.axis.z * g.z.coord(k);
        const double ang = -theta * u;
        psi(i, j, k) =
            psi(i, j, k) * std::complex<double>{std::cos(ang), std::sin(ang)};
    });
}

inline void driven_step(Field3D& psi, const SplitOperator3D& prop, const DipoleDrive& d,
                        double t0, int nsteps) {
    const double dt = prop.dt();
    for (int s = 0; s < nsteps; ++s) {
        const double t = t0 + s * dt;
        apply_dipole_halfkick(psi, d, t, dt);
        prop.step(psi, 1);
        apply_dipole_halfkick(psi, d, t + dt, dt);
    }
}

}  // namespace ses
