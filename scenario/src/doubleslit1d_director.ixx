module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.doubleslit1d_director;
export import ses.scenario.line1d_director;
import ses.interference;
import ses.wavepacket;


// Electron double slit in the TRANSVERSE frame: the 1D axis is the
// coordinate parallel to the slit plane; flight toward the screen is time
// (t = z/v -- the exact paraxial/Fresnel reduction of the real 2D
// experiment). A broad incident wavefront hits the infinite wall with two
// slits at t = 0: the aperture windows it to the openings (the wall
// absorbs the rest; the state is renormalized = post-selection on the
// transmitted electron), then flies free. The far-field screen pattern is
// |psi~(k)|^2 -- FROZEN from the instant of transmission (free flight
// preserves it; ses.interference contract) -- drawn as the fixed cyan
// trace at x = k * T_ref, and the live density band visibly converges
// onto it as t -> T_ref: Young fringes (spacing 2 pi t / d) under the
// single-slit envelope. The flux slider is a solenoid tucked behind the
// wall between the slits: its exact reduced effect multiplies slit 2 by
// e^{i Phi}, sliding the fringes by Phi/d under a FIXED envelope
// (Chambers' Aharonov-Bohm experiment), period one flux quantum 2 pi.


export namespace ses_shell {

constexpr double kDs1dBox = 100.0;
constexpr int kDs1dPoints = 16384;
constexpr double kDs1dDt = 0.05;
constexpr double kDs1dSigmaInc = 30.0;  // incident wavefront width >> d
constexpr double kDs1dSep = 8.0;        // boot slit separation d
constexpr double kDs1dSepMin = 2.0;
constexpr double kDs1dSepMax = 16.0;
constexpr double kDs1dWidth = 1.5;      // boot slit width w
constexpr double kDs1dWidthMin = 0.5;
constexpr double kDs1dWidthMax = 4.0;
constexpr double kDs1dRScale = 40.0;    // live band: radius = 40 |psi|^2
constexpr double kDs1dWallHeight = 25.0;   // display height of the wall
constexpr double kDs1dScreenHeight = 30.0; // display height of the trace

class DoubleSlit1DDirector final : public Line1DDirector, public SlitApi {
public:
    DoubleSlit1DDirector()
        : Line1DDirector(ses::Grid1D{-kDs1dBox, kDs1dBox, kDs1dPoints},
                         std::vector<double>(kDs1dPoints, 0.0), kDs1dDt,
                         kDs1dRScale, 1.0, 1e30) {
        // Soft absorber at the box edges: the expanding pattern leaves the
        // stage instead of wrapping the periodic grid back through it.
        set_mask(ses::absorbing_mask(grid1d_, 12.0));
        fire();
    }

    SlitApi* slit() override { return this; }

    // ---- SlitApi ----
    void set_separation(double d) override {
        sep_ = std::clamp(d, kDs1dSepMin, kDs1dSepMax);
        fire();
    }
    double separation() const override { return sep_; }
    void set_width(double w) override {
        width_ = std::clamp(w, kDs1dWidthMin, kDs1dWidthMax);
        fire();
    }
    double width() const override { return width_; }
    void set_flux(double phi) override {
        flux_ = phi;
        fire();
    }
    double flux() const override { return flux_; }
    void refire() override { fire(); }
    double transmitted_fraction() const override { return transmitted_; }
    double screen_at(double k) const override {
        if (screen_axis_.empty()) {
            return 0.0;
        }
        std::size_t best = 0;
        for (std::size_t j = 1; j < screen_axis_.size(); ++j) {
            if (std::abs(screen_axis_[j] - k) <
                std::abs(screen_axis_[best] - k)) {
                best = j;
            }
        }
        return screen_spec_[best];
    }

    bool handle_key(char key) override {
        if (key == '2') {
            fire();
            return true;
        }
        return false;
    }

    double default_camera_azimuth() const override { return 0.25; }
    double default_camera_elevation() const override { return 0.22; }
    double default_camera_distance() const override { return 60.0; }

    // Base curves 0..3 (sheet, band, red profile, white phasor) with the
    // red slot REPLACED by the wall-with-slits face, plus the fixed screen
    // trace on top.
    int overlay_curve_count() const override { return 5; }
    OverlayCurve overlay_curve(int i) const override {
        if (i == 2) {  // the pierced wall, face on
            return {wall_curve_.data(),
                    static_cast<int>(wall_curve_.size() / 3),
                    1.0f, 0.30f, 0.25f, 0.9f};
        }
        if (i == 4) {  // far-field screen: fixed from the aperture instant
            return {screen_curve_.data(),
                    static_cast<int>(screen_curve_.size() / 3),
                    0.35f, 0.85f, 1.0f, 0.95f};
        }
        return Line1DDirector::overlay_curve(i);
    }

protected:
    const char* scene_name() const override {
        return "Electron double slit (transverse frame)";
    }

    std::string title_suffix() override {
        return strf("  d = %.1f  w = %.1f  Phi = %.2f pi  T = %.1f%%  "
                    "screen t_ref = %.1f (fringe %.2f Bohr)  t/t_ref = %.2f"
                    "  keys: 2 refire",
                    sep_, width_, flux_ / 3.14159265358979323846,
                    100.0 * transmitted_, t_ref_,
                    2.0 * 3.14159265358979323846 * t_ref_ / sep_,
                    t_ref_ > 0.0 ? sim_time_ / t_ref_ : 0.0);
    }

private:
    // One electron through the wall: broad wavefront, aperture, free
    // flight from t = 0. The screen trace is computed HERE, once -- it is
    // invariant from this instant on.
    void fire() {
        ses::Field1D psi =
            ses::gaussian_wavepacket(grid1d_, 0.0, kDs1dSigmaInc, 0.0);
        const std::vector<std::complex<double>> ap =
            ses::double_slit_aperture(grid1d_, sep_, width_, flux_);
        transmitted_ = ses::apply_aperture(psi, ap);
        sim_time_ = 0.0;
        pending_steps_ = 0;
        set_state(std::move(psi));  // live state AND the reset target
        rebuild_wall();
        rebuild_screen();
        note_refresh();
    }

    void rebuild_wall() {
        // Display-only face of the infinite wall (the PHYSICS potential
        // stays zero: the wall acted once, as the aperture).
        std::vector<double> v(static_cast<std::size_t>(grid1d_.n),
                              kDs1dWallHeight);
        for (int i = 0; i < grid1d_.n; ++i) {
            const double x = grid1d_.coord(i);
            if (std::abs(x + 0.5 * sep_) <= 0.5 * width_ ||
                std::abs(x - 0.5 * sep_) <= 0.5 * width_) {
                v[static_cast<std::size_t>(i)] = 0.0;
            }
        }
        wall_curve_ = ses::potential_curve(grid1d_, v, 1.0, 1e30);
    }

    void rebuild_screen() {
        screen_spec_ = ses::momentum_spectrum(psi_);
        screen_axis_ = ses::momentum_axis(grid1d_);
        // Angle mapping x = k * t_ref, with t_ref sized so twice the
        // single-slit envelope (|k| < 2 pi / w) spans ~90% of the box; the
        // height is display-normalized to a fixed peak.
        t_ref_ = 0.9 * kDs1dBox * width_ /
                 (2.0 * 2.0 * 3.14159265358979323846);
        double peak = 0.0;
        for (const double s : screen_spec_) {
            peak = std::max(peak, s);
        }
        const double ys = peak > 0.0 ? kDs1dScreenHeight / peak : 0.0;
        screen_curve_.clear();
        for (std::size_t j = 0; j < screen_axis_.size(); ++j) {
            const double x = screen_axis_[j] * t_ref_;
            if (std::abs(x) > kDs1dBox) {
                continue;
            }
            screen_curve_.push_back(static_cast<float>(x));
            screen_curve_.push_back(
                static_cast<float>(ys * screen_spec_[j]));
            screen_curve_.push_back(0.0f);
        }
    }

    void note_refresh() {
        display_changed_ = true;
        title_dirty_ = true;
    }

    double sep_ = kDs1dSep;
    double width_ = kDs1dWidth;
    double flux_ = 0.0;
    double transmitted_ = 0.0;
    double t_ref_ = 0.0;
    std::vector<double> screen_spec_;
    std::vector<double> screen_axis_;
    std::vector<float> screen_curve_;
    std::vector<float> wall_curve_;
};

}  // namespace ses_shell
