module;
#include <cmath>
#include <string>
export module ses.scenario.harmonic1d_director;
export import ses.scenario.line1d_director;
import ses.ladder;
import ses.wavepacket;


// Textbook 1D harmonic oscillator with ladder-operator controls. The ground
// state is the exact Gaussian (sigma = 1/sqrt(2 omega)); [U] applies a-dag,
// [D] applies a. Down from the ground state is refused by the operator
// itself: a|0> = 0, so ladder_lower reports a vanishing norm and leaves psi
// untouched -- the refusal is physics, not a UI rule. Up is capped at the
// spectral-band level where the FFT ladder stays clean on this grid (the
// grid Nyquist is matched to the physics band; see tests/ladder_test.cpp).
// Stationary states show phase as rigid rotation of the phasor curve about
// the x axis at rate E_n = (n + 1/2) omega.


export namespace ses_shell {

constexpr double kHo1dOmega = 0.25;   // matches the 3D trap's kTrapOmega
constexpr double kHo1dBox = 20.0;     // Bohr half-extent
// 256 points: k_max ~ 20 vs the n<=10 band (~2.5) -- the ladder noise gain
// k_max/sqrt(2 omega) stays low enough for a clean chain (ladder_test).
constexpr int kHo1dPoints = 256;
constexpr double kHo1dDt = 0.04;
constexpr int kHo1dMaxLevel = 10;
constexpr double kHo1dRScale = 18.0;  // radius = 18 |psi|^2 (~5 Bohr at n=0)
constexpr double kHo1dEScale = 0.8;   // V display: Ha -> Bohr height
constexpr double kHo1dYClamp = 10.0;  // parabola leaves the frame at +-10

class Harmonic1DDirector final : public Line1DDirector, public Ladder1dApi {
public:
    Harmonic1DDirector()
        : Line1DDirector(ses::Grid1D{-kHo1dBox, kHo1dBox, kHo1dPoints},
                         ses::harmonic_potential(
                             ses::Grid1D{-kHo1dBox, kHo1dBox, kHo1dPoints},
                             kHo1dOmega),
                         kHo1dDt, kHo1dRScale, kHo1dEScale, kHo1dYClamp) {
        set_state(ground());
    }

    Ladder1dApi* ladder1d() override { return this; }

    // ---- Ladder1dApi ----
    int level() const override { return level_; }
    double level_energy() const override {
        return ses::mean_energy(psi_, potential_);
    }
    bool ladder(bool up) override {
        if (up) {
            if (level_ >= kHo1dMaxLevel) {
                note_ = strf("n = %d cap (spectral band)", kHo1dMaxLevel);
                title_dirty_ = true;
                return false;
            }
            // ||adag psi_n||^2 = n + 1 -> the new level, counted by the
            // operator itself.
            const double norm2 = ses::ladder_raise(psi_, kHo1dOmega);
            level_ = static_cast<int>(std::lround(norm2));
            note_.clear();
        } else {
            // ||a psi_n||^2 = n; ~0 means annihilation (psi untouched).
            const double norm2 = ses::ladder_lower(psi_, kHo1dOmega);
            if (norm2 < 0.5) {
                note_ = "a|0> = 0: refused";
                title_dirty_ = true;
                return false;
            }
            level_ = static_cast<int>(std::lround(norm2)) - 1;
            note_.clear();
        }
        mark_display_dirty();
        title_dirty_ = true;
        return true;
    }

    bool handle_key(char key) override {
        if (key == 'U') {
            ladder(true);
            return true;
        }
        if (key == 'D') {
            ladder(false);
            return true;
        }
        if (key == '2') {
            reset_simulation();
            return true;
        }
        return false;
    }

    double default_camera_azimuth() const override { return 0.35; }
    double default_camera_elevation() const override { return 0.28; }
    double default_camera_distance() const override { return 55.0; }

protected:
    const char* scene_name() const override { return "1D harmonic ladder"; }

    std::string title_suffix() override {
        const double e = ses::mean_energy(psi_, potential_);
        std::string s = strf(
            "  w = %.2f  n = %d  <H> = %.4f Ha ((n+1/2)w = %.4f)", kHo1dOmega,
            level_, e, (level_ + 0.5) * kHo1dOmega);
        if (!note_.empty()) {
            s += "  [" + note_ + "]";
        }
        s += "  keys: U up / D down / 2 ground";
        return s;
    }

    void after_reset() override {
        level_ = 0;
        note_.clear();
    }

private:
    ses::Field1D ground() const {
        // sigma = 1/sqrt(2 omega): the exact HO ground state.
        return ses::gaussian_wavepacket(grid1d_, 0.0,
                                        1.0 / std::sqrt(2.0 * kHo1dOmega), 0.0);
    }

    int level_ = 0;
    std::string note_;
};

}  // namespace ses_shell
