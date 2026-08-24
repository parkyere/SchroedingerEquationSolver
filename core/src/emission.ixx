module;
#include <numbers>
#include <algorithm>

#include <cmath>
#include <complex>
#include <cstddef>
#include <span>
#include <vector>
export module ses.emission;
export import ses.decay;
export import ses.field;
import ses.harmonics;
import ses.vec;
import ses.grid;


// Semiclassical (Larmor) dipole emission + QED photon record (direction,
// helicity) conditioning, atomic units. Larmor: coherent-superposition only,
// P == 0 for a pure eigenstate (its decay = Einstein-A jumps, ses.decay).
// P = (2/3) alpha^3 |d_ddot|^2, d_ddot = <grad V> (Ehrenfest).


export namespace ses {

// PRECONDITION: psi normalized; result is the raw integral (NOT norm-invariant),
// so unnormalized psi scales it by the norm. GPU mean_force oracle shares this.
inline Vec3d mean_potential_gradient(const Field3D& psi, const std::vector<double>& v,
                                     const Grid3D& g) noexcept {
    const int nx = g.x.n;
    const int ny = g.y.n;
    const int nz = g.z.n;
    const double inv2hx = 1.0 / (2.0 * g.x.spacing());
    const double inv2hy = 1.0 / (2.0 * g.y.spacing());
    const double inv2hz = 1.0 / (2.0 * g.z.spacing());
    Vec3d acc{};
    for (int k = 0; k < nz; ++k) {
        const int kp = (k + 1) % nz;
        const int km = (k - 1 + nz) % nz;
        for (int j = 0; j < ny; ++j) {
            const int jp = (j + 1) % ny;
            const int jm = (j - 1 + ny) % ny;
            for (int i = 0; i < nx; ++i) {
                const int ip = (i + 1) % nx;
                const int im = (i - 1 + nx) % nx;
                const double rho = std::norm(psi(i, j, k));
                const double gx =
                    (v[static_cast<std::size_t>(g.flat(ip, j, k))] -
                     v[static_cast<std::size_t>(g.flat(im, j, k))]) * inv2hx;
                const double gy =
                    (v[static_cast<std::size_t>(g.flat(i, jp, k))] -
                     v[static_cast<std::size_t>(g.flat(i, jm, k))]) * inv2hy;
                const double gz =
                    (v[static_cast<std::size_t>(g.flat(i, j, kp))] -
                     v[static_cast<std::size_t>(g.flat(i, j, km))]) * inv2hz;
                acc.x += rho * gx;
                acc.y += rho * gy;
                acc.z += rho * gz;
            }
        }
    }
    const double dv = g.cell_volume();
    return Vec3d{acc.x * dv, acc.y * dv, acc.z * dv};
}

inline constexpr double larmor_power(const Vec3d& dipole_accel) noexcept {
    const double a3 = kFineStructureConstant * kFineStructureConstant *
                      kFineStructureConstant;
    return (2.0 / 3.0) * a3 * dot(dipole_accel, dipole_accel);
}

// ---- QED photon record: E1 direction + helicity conditioning ----
// Detecting the photon as a plane wave (n, lambda) projects the atom onto
// c_m ~ conj(e_lambda(n)) . D_m over the degenerate destination sublevels;
// angular momentum bookkeeping is automatic in this coupling.

struct PhotonRecord {
    Vec3d n;       // propagation direction (unit)
    int helicity;  // +1 / -1 along n
};

// e_lambda(n) = (theta_hat + i lambda phi_hat)/sqrt2; e_+(z_hat) =
// (x_hat + i y_hat)/sqrt2 pins the convention (sigma+ along +z). n unit.
inline DipoleMatrixElement helicity_vector(const Vec3d& n, int lambda) noexcept {
    const double st = std::sqrt(std::max(0.0, n.x * n.x + n.y * n.y));
    const double cph = st > 0.0 ? n.x / st : 1.0;  // poles: phi = 0
    const double sph = st > 0.0 ? n.y / st : 0.0;
    const double ct = n.z;
    const Vec3d th{ct * cph, ct * sph, -st};
    const Vec3d ph{-sph, cph, 0.0};
    const double s2 = std::sqrt(0.5);
    const double l = lambda >= 0 ? 1.0 : -1.0;
    return DipoleMatrixElement{{th.x * s2, l * ph.x * s2},
                               {th.y * s2, l * ph.y * s2},
                               {th.z * s2, l * ph.z * s2}};
}

// c_m ~ conj(e_lambda(n)) . D_m, normalized; all-zero dipoles stay all-zero.
inline std::vector<std::complex<double>> conditioned_amplitudes(
    const std::vector<DipoleMatrixElement>& dipoles, const Vec3d& n,
    int lambda) {
    const DipoleMatrixElement e = helicity_vector(n, lambda);
    std::vector<std::complex<double>> c(dipoles.size());
    double n2 = 0.0;
    for (std::size_t m = 0; m < dipoles.size(); ++m) {
        const DipoleMatrixElement& d = dipoles[m];
        c[m] = std::conj(e.x) * d.x + std::conj(e.y) * d.y +
               std::conj(e.z) * d.z;
        n2 += std::norm(c[m]);
    }
    if (n2 > 0.0) {
        const double inv = 1.0 / std::sqrt(n2);
        for (std::complex<double>& z : c) {
            z *= inv;
        }
    }
    return c;
}

// Angular dipole vectors <to|r|from> across ONE destination shell (fixed
// l_to; the common radial integral is omitted -- it cancels in conditioning
// and in direction sampling). Signs from tesseral_e1_axis.
inline std::vector<DipoleMatrixElement> shell_dipole_vectors(
    int l_from, int m_from, int l_to, const std::vector<int>& m_to) {
    std::vector<DipoleMatrixElement> d(m_to.size());
    for (std::size_t i = 0; i < m_to.size(); ++i) {
        d[i] = DipoleMatrixElement{
            {tesseral_e1_axis(0, l_to, m_to[i], l_from, m_from), 0.0},
            {tesseral_e1_axis(1, l_to, m_to[i], l_from, m_from), 0.0},
            {tesseral_e1_axis(2, l_to, m_to[i], l_from, m_from), 0.0}};
    }
    return d;
}

// ---- collective E1 jump (equal-gap ladders) -------------------------------
// When every transition shares ONE frequency (harmonic ladder), the photon
// cannot resolve which rung fired: the jump operator is the collective
// lowering sum (damped-oscillator L ~ a per axis). Building the
// per-destination dipole vectors D_to = sum_ch m_ch c_from reduces the jump
// to the EXISTING machinery: sample_photon_emission({D_to}) picks (n,lambda),
// conditioned_amplitudes({D_to}) is the post-jump superposition over
// to_states -- coherences between rungs survive.

struct DipoleChannel {
    int from;
    int to;
    double mx, my, mz;  // signed <to|r_axis|from> (au)
};

struct CollectiveDipoles {
    std::vector<int> to_states;                // unique destinations, in order
    std::vector<DipoleMatrixElement> dipoles;  // D_to (parallel to to_states)
};

inline CollectiveDipoles collective_jump_dipoles(
    const std::vector<DipoleChannel>& channels,
    std::span<const std::complex<double>> c) {
    CollectiveDipoles out;
    for (const DipoleChannel& ch : channels) {
        const std::complex<double> a = c[static_cast<std::size_t>(ch.from)];
        std::size_t slot = out.to_states.size();
        for (std::size_t i = 0; i < out.to_states.size(); ++i) {
            if (out.to_states[i] == ch.to) {
                slot = i;
                break;
            }
        }
        if (slot == out.to_states.size()) {
            out.to_states.push_back(ch.to);
            out.dipoles.push_back(DipoleMatrixElement{});
        }
        DipoleMatrixElement& d = out.dipoles[slot];
        d.x += ch.mx * a;
        d.y += ch.my * a;
        d.z += ch.mz * a;
    }
    return out;
}

// ---- frequency-resolved collective decay (one accumulated interval) -------
// The environment measures the photon FREQUENCY wherever line separations
// exceed widths: channels cluster into frequency groups, each an independent
// collective jump operator. Equal-gap ladders collapse to ONE group (the
// damped-HO limit); l-degenerate hydrogen shells share a group (3p->2s and
// 3d->2p interfere); distinct n-gaps do not.

struct GroupedChannel {
    DipoleChannel ch;
    double gap_e;   // transition energy (photon bookkeeping)
    double k_rate;  // display-rate scale: gamma_ch / |m_ch|^2
};

struct FreqGroup {
    std::vector<DipoleChannel> channels;
    double gap_e = 0.0;
    double k_rate = 0.0;
};

// Cluster by gap (ascending); a new group starts when the gap exceeds the
// running group's first gap by more than tol.
inline std::vector<FreqGroup> group_by_gap(std::vector<GroupedChannel> items,
                                           double tol) {
    (void)items;
    (void)tol;
    return {};  // stub (red)
}

struct IntervalJump {
    PhotonRecord rec;
    double gap_e = 0.0;
    int dominant_to = -1;
};

struct IntervalResult {
    std::vector<IntervalJump> jumps;
    // Post-interval amplitudes; equals the input when no jump fired.
    std::vector<std::complex<double>> c;
};

// One accumulated-interval unraveling over frequency groups. Per arrival the
// uniform stream is consumed as: (1) arrival time u, (2) group-pick u
// (stratified by group rate, drawn even for a single group), then
// sample_photon_emission's draws. Each jump re-conditions c coherently over
// the fired group's destinations; chains continue analytically.
template <class U01>
inline IntervalResult collective_decay_interval(
    const std::vector<FreqGroup>& groups, std::vector<std::complex<double>> c,
    double dt, U01&& u01) {
    (void)groups;
    (void)dt;
    (void)u01;
    return IntervalResult{{}, std::move(c)};  // stub (red)
}

// Joint (n, lambda) sample from P ~ Sum_m |conj(e_lambda(n)) . D_m|^2 by
// rejection against the bound Sum_m |D_m|^2 (transverse projector never
// exceeds it); u01() supplies uniforms in [0,1).
template <class U01>
inline PhotonRecord sample_photon_emission(
    const std::vector<DipoleMatrixElement>& dipoles, U01&& u01) {
    double bound = 0.0;
    for (const DipoleMatrixElement& d : dipoles) {
        bound += std::norm(d.x) + std::norm(d.y) + std::norm(d.z);
    }
    if (bound <= 0.0) {
        return PhotonRecord{Vec3d{0.0, 0.0, 1.0}, +1};  // forbidden guard
    }
    constexpr double kPi = std::numbers::pi;
    for (;;) {
        const double ct = 1.0 - 2.0 * u01();
        const double st = std::sqrt(std::max(0.0, 1.0 - ct * ct));
        const double phi = 2.0 * kPi * u01();
        const Vec3d n{st * std::cos(phi), st * std::sin(phi), ct};
        double w[2];  // lambda = +1, -1
        for (int i = 0; i < 2; ++i) {
            const DipoleMatrixElement e = helicity_vector(n, i == 0 ? +1 : -1);
            double s = 0.0;
            for (const DipoleMatrixElement& d : dipoles) {
                s += std::norm(std::conj(e.x) * d.x + std::conj(e.y) * d.y +
                               std::conj(e.z) * d.z);
            }
            w[i] = s;
        }
        const double f = w[0] + w[1];
        if (u01() * bound < f) {
            return PhotonRecord{n, u01() * f < w[0] ? +1 : -1};
        }
    }
}

}  // namespace ses
