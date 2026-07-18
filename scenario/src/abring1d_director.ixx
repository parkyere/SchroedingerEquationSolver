module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.abring1d_director;
export import ses.scenario.line1d_director;


// Aharonov-Bohm on a 1D ring. The periodic FFT grid IS a ring; the
// display unrolls it (the box edges are one and the same point). A flux
// Phi threads the hole: H = (p - A)^2 / 2 with A = Phi / L uniform, so
// B = dA/dx = 0 everywhere ON the ring -- locally there is no force, yet
// two wavepacket halves sent around opposite sides pick up a relative
// phase of EXACTLY Phi when they recombine at the antipode. The pair is
// injected with mechanical momentum +-k0 (canonical k = +-k0 + A, equal
// kinetic energy -- as a real ring interferometer injects), so both
// halves arrive together and the meeting fringe pattern shifts with the
// flux: bright at Phi = 0, dark at Phi = pi, period one flux quantum
// (2 pi in a.u.). The phase winding e^{iAx} is visible on the live band
// as a uniform hue gradient with NO density change: potential without
// force, the AB signature.


export namespace ses_shell {

constexpr double kAb1dBox = 50.0;   // ring circumference L = 100 Bohr
constexpr int kAb1dPoints = 4096;
constexpr double kAb1dDt = 0.01;
constexpr double kAb1dSigma = 4.0;  // packet envelope width
constexpr int kAb1dK0Bin = 16;      // mechanical momentum: bin 16 ~ 1.0
constexpr double kAb1dX0 = -25.0;   // launch point (-L/4)
constexpr double kAb1dRScale = 55.0;

class RingAB1DDirector final : public Line1DDirector, public RingApi {
public:
    RingAB1DDirector()
        : Line1DDirector(ses::Grid1D{-kAb1dBox, kAb1dBox, kAb1dPoints},
                         std::vector<double>(kAb1dPoints, 0.0), kAb1dDt,
                         kAb1dRScale, 1.0, 1e30) {
        fire();
    }

    RingApi* ring() override { return this; }

    // ---- RingApi ----
    void set_flux(double phi) override {
        flux_ = phi;
        // A rides in the kinetic tables; a new flux is a new injection.
        a_field_ = flux_ / circumference();
        set_potential(std::vector<double>(
            static_cast<std::size_t>(grid1d_.n), 0.0));
        fire();
    }
    double flux() const override { return flux_; }
    void refire() override { fire(); }
    double meet_time() const override {
        return 0.5 * circumference() / k0();
    }
    double meet_density() const override {
        // Crest-averaged |psi|^2 within 0.15 fringe periods (pi / k0) of
        // the meeting point -- same probe the core contract test uses.
        const double xm = meet_point();
        const double win = 0.15 * 3.14159265358979323846 / k0();
        double sum = 0.0;
        int cnt = 0;
        for (int i = 0; i < grid1d_.n; ++i) {
            if (std::abs(grid1d_.coord(i) - xm) < win) {
                sum += std::norm(psi_[i]);
                ++cnt;
            }
        }
        return cnt > 0 ? sum / cnt : 0.0;
    }

    double meet_density_max() const override { return meet_max_; }

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

    // Base curves with the red potential slot (V = 0: a bare axis line)
    // REPLACED by the meeting-point marker.
    OverlayCurve overlay_curve(int i) const override {
        if (i == 2) {
            return {meet_marker_.data(), 2, 1.0f, 0.30f, 0.25f, 0.9f};
        }
        return Line1DDirector::overlay_curve(i);
    }

protected:
    const char* scene_name() const override {
        return "Aharonov-Bohm ring (unrolled)";
    }

    void after_batch() override {
        meet_max_ = std::max(meet_max_, meet_density());
    }

    std::string title_suffix() override {
        const double pi = 3.14159265358979323846;
        return strf("  Phi = %.2f flux quanta  k0 = %.2f  meet at x = %.0f, "
                    "t = %.1f au  density(meet) = %.4f  keys: 2 refire",
                    flux_ / (2.0 * pi), k0(), meet_point(), meet_time(),
                    meet_density());
    }

private:
    double circumference() const { return grid1d_.xmax - grid1d_.xmin; }
    double k0() const {
        return kAb1dK0Bin * 2.0 * 3.14159265358979323846 / circumference();
    }
    double meet_point() const { return kAb1dX0 + 0.5 * circumference(); }

    // Inject the +-k0 mechanical-momentum pair at the launch point.
    void fire() {
        ses::Field1D psi{grid1d_};
        const double a = a_field_;
        for (int i = 0; i < grid1d_.n; ++i) {
            const double x = grid1d_.coord(i);
            const double u = x - kAb1dX0;
            const double env =
                std::exp(-u * u / (2.0 * kAb1dSigma * kAb1dSigma));
            const std::complex<double> boost{std::cos(a * x),
                                             std::sin(a * x)};
            psi[i] = env * boost * 2.0 * std::cos(k0() * u);
        }
        ses::normalize(psi);
        sim_time_ = 0.0;
        pending_steps_ = 0;
        meet_max_ = 0.0;
        set_state(std::move(psi));
        const float xm = static_cast<float>(meet_point());
        meet_marker_ = {xm, -8.0f, 0.0f, xm, 8.0f, 0.0f};
        display_changed_ = true;
        title_dirty_ = true;
    }

    double flux_ = 0.0;
    double meet_max_ = 0.0;
    std::vector<float> meet_marker_;
};

}  // namespace ses_shell
