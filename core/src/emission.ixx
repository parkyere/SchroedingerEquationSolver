module;
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
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

// ---- photon streak display -----------------------------------------------
// Helix e^{i lambda k s} around the flight axis (1D phasor grammar: twist
// sense = helicity); wall-clock flight, ~2 s to exit the box at 60 fps.

inline constexpr int kPhotonFlightTicks = 120;      // ~2 s at 60 fps
inline constexpr double kPhotonTravel = 160.0;      // Bohr; exits the +-80 box
inline constexpr double kPhotonWaveScale = 9.375;   // lambda = scale/dE (Lyman-alpha -> 25)
inline constexpr double kPhotonStreakRadius = 4.0;  // Bohr
inline constexpr int kPhotonStreakPoints = 96;      // body + tip vertex

inline constexpr double kPhotonTipLen = 8.0;  // Bohr; on-axis arrowhead reach
inline constexpr double kPhotonTurns = 4.0;   // visible tail = 4 wavelengths

inline constexpr double kPhotonRefDeltaE = 0.375;  // Lyman-alpha baseline

// Display wavelength: relative energies honest (true lambda >> box).
inline double photon_display_wavelength(double delta_e) noexcept {
    return delta_e > 0.0 ? kPhotonWaveScale / delta_e : kPhotonTravel;
}

// Flight distance: base box-exit travel + the 4-wavelength tail unspools
// fully before the fade completes.
inline double photon_travel(double delta_e) noexcept {
    return kPhotonTravel + kPhotonTurns * photon_display_wavelength(delta_e);
}

// Flight lifetime in frames: Lyman-alpha (2p->1s) = kPhotonFlightTicks (2 s),
// every other photon flies at the SAME display speed (frames scale with
// travel).
inline int photon_flight_frames(double delta_e) noexcept {
    return static_cast<int>(
        std::lround(kPhotonFlightTicks * photon_travel(delta_e) /
                    photon_travel(kPhotonRefDeltaE)));
}

// Photon flash: linear decay, peak capped at 1/3 (full bright hurt the eye).
inline constexpr double kFlashPeak = 1.0 / 3.0;

inline double flash_intensity(int ticks_left, int ticks_total) noexcept {
    // stub (red): today's full-bright ramp
    return ticks_total > 0 ? static_cast<double>(ticks_left) /
                                 static_cast<double>(ticks_total)
                           : 0.0;
}

// Up to kMaxPhotonFlights concurrent streaks; a spawn beyond capacity evicts
// the most-spent flight (largest progress).
inline constexpr int kMaxPhotonFlights = 4;

struct PhotonFlightPool {
    struct Flight {
        PhotonRecord rec{};
        double delta_e = 0.0;
        int age_frames = -1;  // -1 = free slot
        int total_frames = 1;
    };
    std::array<Flight, kMaxPhotonFlights> slots{};

    // stub (red): single-slot semantics (today's behavior)
    void spawn(const PhotonRecord& rec, double delta_e, int total_frames) {
        slots[0] = Flight{rec, delta_e, 0, total_frames};
    }
    void advance() {
        if (slots[0].age_frames >= 0 &&
            ++slots[0].age_frames > slots[0].total_frames) {
            slots[0].age_frames = -1;
        }
    }
    int count() const {
        return slots[0].age_frames >= 0 ? 1 : 0;
    }
    // i-th ACTIVE flight in slot order; nullptr past count().
    const Flight* active(int i) const {
        return i == 0 && slots[0].age_frames >= 0 ? &slots[0] : nullptr;
    }
};

// 1 through most of the flight, linear fade to 0 at the end.
inline double photon_streak_alpha(double progress) noexcept {
    constexpr double kFadeStart = 0.7;
    const double p = std::clamp(progress, 0.0, 1.0);
    return p <= kFadeStart ? 1.0 : (1.0 - p) / (1.0 - kFadeStart);
}

// Body helix trails the head by kPhotonTurns wavelengths (clamped at the
// nucleus); the last vertex is the on-axis arrow tip ahead of the body.
// Transverse frame = helicity_vector's (theta_hat, phi_hat) and phase =
// k(s - s_head), so the spring twists AND rotates in flight, sense = helicity.
inline std::vector<Vec3d> photon_streak_vertices(const PhotonRecord& ph,
                                                 double delta_e,
                                                 double progress) {
    const Vec3d n = ph.n;
    const double st = std::sqrt(std::max(0.0, n.x * n.x + n.y * n.y));
    const double cph = st > 0.0 ? n.x / st : 1.0;  // poles: phi = 0
    const double sph = st > 0.0 ? n.y / st : 0.0;
    const Vec3d th{n.z * cph, n.z * sph, -st};
    const Vec3d fi{-sph, cph, 0.0};
    const double lam_len = photon_display_wavelength(delta_e);
    const double sh =
        std::clamp(progress, 0.0, 1.0) * photon_travel(delta_e);
    const double s0 = std::max(0.0, sh - kPhotonTurns * lam_len);
    constexpr double kPi = 3.14159265358979323846;
    const double k = 2.0 * kPi / lam_len;
    const double lam = ph.helicity >= 0 ? 1.0 : -1.0;
    std::vector<Vec3d> v(static_cast<std::size_t>(kPhotonStreakPoints));
    const int nb = kPhotonStreakPoints - 1;
    for (int i = 0; i < nb; ++i) {
        const double s =
            s0 + (sh - s0) * static_cast<double>(i) / (nb - 1);
        const double a = k * (s - sh);
        const double cx = std::cos(a) * kPhotonStreakRadius;
        const double cy = lam * std::sin(a) * kPhotonStreakRadius;
        v[static_cast<std::size_t>(i)] =
            Vec3d{n.x * s + th.x * cx + fi.x * cy,
                  n.y * s + th.y * cx + fi.y * cy,
                  n.z * s + th.z * cx + fi.z * cy};
    }
    const double tip = sh + kPhotonTipLen;
    v[static_cast<std::size_t>(nb)] =
        Vec3d{n.x * tip, n.y * tip, n.z * tip};
    return v;
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
    constexpr double kPi = 3.14159265358979323846;
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
