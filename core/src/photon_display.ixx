module;
#include <numbers>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>
export module ses.photon_display;
export import ses.emission;
import ses.vec;


// Photon DISPLAY policy: streak helix e^{i lambda k s} (twist/rotation sense
// = helicity, comoving phase), constant display speed (Lyman-alpha = 2 s),
// flash peak, concurrent-flight pool. The physics (conditioning, sampling)
// lives in ses.emission.


export namespace ses {

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
    if (ticks_total <= 0 || ticks_left <= 0) {
        return 0.0;
    }
    return kFlashPeak * static_cast<double>(ticks_left) /
           static_cast<double>(ticks_total);
}

// Concurrent streak capacity: a 5-photon Yrast cascade (6h->...->1s) times
// three overlapping bursts fits; ~= emission rate x the 48 s softest-line
// life. Renderer holds 64 overlay slots, so eviction is a safety valve only.
inline constexpr int kMaxPhotonFlights = 16;

struct PhotonFlightPool {
    struct Flight {
        PhotonRecord rec{};
        double delta_e = 0.0;
        int age_frames = -1;  // -1 = free slot
        int total_frames = 1;
    };
    std::array<Flight, kMaxPhotonFlights> slots{};

    // Free slot first; else evict the most-spent flight (largest progress).
    void spawn(const PhotonRecord& rec, double delta_e, int total_frames) {
        int pick = 0;
        double worst = -1.0;
        for (int i = 0; i < kMaxPhotonFlights; ++i) {
            const Flight& f = slots[static_cast<std::size_t>(i)];
            if (f.age_frames < 0) {
                pick = i;
                break;
            }
            const double spent = static_cast<double>(f.age_frames) /
                                 static_cast<double>(f.total_frames);
            if (spent > worst) {
                worst = spent;
                pick = i;
            }
        }
        slots[static_cast<std::size_t>(pick)] =
            Flight{rec, delta_e, 0, total_frames < 1 ? 1 : total_frames};
    }
    // Ages every active flight; expires past its own total.
    void advance() {
        for (Flight& f : slots) {
            if (f.age_frames < 0) {
                continue;
            }
            ++f.age_frames;
            if (f.age_frames > f.total_frames) {
                f.age_frames = -1;
            }
        }
    }
    int count() const {
        int n = 0;
        for (const Flight& f : slots) {
            n += f.age_frames >= 0 ? 1 : 0;
        }
        return n;
    }
    // i-th ACTIVE flight in slot order; nullptr past count().
    const Flight* active(int i) const {
        for (const Flight& f : slots) {
            if (f.age_frames < 0) {
                continue;
            }
            if (i == 0) {
                return &f;
            }
            --i;
        }
        return nullptr;
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
    constexpr double kPi = std::numbers::pi;
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

}  // namespace ses
