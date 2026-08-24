module;
#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
export module ses.scenario.tunneling_director;
export import ses.scenario.base_director;

export namespace ses_shell {

constexpr double kTunnelBox = 80.0;      // Bohr; h = 0.625 (256^3)
constexpr int kTunnelPoints = 256;
constexpr double kTunnelDt = 0.04;
constexpr double kTunnelV0 = 0.25;       // Ha
// kappa = sqrt(2(V0-E)) = 0.5; analytic T = [1 + sinh^2(kappa w)]^-1 ~ 2.7%
constexpr double kTunnelXLo = 0.0;
constexpr double kTunnelXHi = 5.0;
constexpr double kTunnelK0 = 0.5;        // mean E = k^2/2 = 0.125 < V0
constexpr double kTunnelLaunchX = -30.0;
constexpr double kTunnelSigma = 5.0;

class TunnelingDirector final : public BaseDirector, public TunnelApi {
public:
    TunnelingDirector() : BaseDirector(make()) {}

    TunnelApi* tunnel() override { return this; }

    // Selftest hook.
    double transmitted_max() const override { return t_max_; }

protected:
    ses::WavepacketSimulation remake_simulation() const override { return make(); }
    const char* scene_name() const override { return "Quantum tunneling"; }
    double absorber_width() const override { return 10.0; }
    bool relax_allowed() const override { return false; }  // no bound target

    // No marker: an origin ball would misread as a nucleus; renderer draws the slab.
    int marker_count() const override { return 0; }
    std::optional<Slab> barrier_slab() const override {
        return Slab{kTunnelXLo, kTunnelXHi};
    }
    // Slight angle keeps the 3D depth cue on the wall.
    double default_camera_azimuth() const override { return 0.18; }
    double default_camera_elevation() const override { return 0.22; }

    std::string title_suffix() override {
        return strf("  V0 = {:.2f} eV, E = {:.2f} eV (forbidden)  P(x<{:.0f}) {:.3f} | "
                    "P(x>{:.0f}) {:.3f} (max T {:.3f})",
                    kTunnelV0 * kHaToEv,
                    0.5 * kTunnelK0 * kTunnelK0 * kHaToEv, kTunnelXLo,
                    p_left_, kTunnelXHi, p_right_, t_max_);
    }

    // Slab transmission tallies via the base probe (gate/cadence/skip there).
    void after_step_batch() override {
        const ses::Grid3D& g = sim_.grid();
        const double dv = g.cell_volume();
        double left = 0.0;
        double right = 0.0;
        if (!probe_readback([&](int i, int /*j*/, int /*k*/, double d2) {
                const double d = d2 * dv;
                const double x = g.x.coord(i);
                if (x < kTunnelXLo) {
                    left += d;
                } else if (x >= kTunnelXHi) {
                    right += d;
                }
            })) {
            return;
        }
        p_left_ = left;
        p_right_ = right;
        t_max_ = std::max(t_max_, p_right_);
        title_dirty_ = true;
    }

    void after_reset() override {
        p_left_ = 0.0;
        p_right_ = 0.0;
        t_max_ = 0.0;
    }

private:
    static ses::WavepacketSimulation make() {
        const ses::Grid1D axis{-kTunnelBox, kTunnelBox, kTunnelPoints};
        const ses::Grid3D grid{axis, axis, axis};
        return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
            grid,
            ses::barrier_potential(grid, kTunnelV0, kTunnelXLo, kTunnelXHi),
            ses::Vec3d{kTunnelLaunchX, 0.0, 0.0},
            ses::Vec3d{kTunnelSigma, kTunnelSigma, kTunnelSigma},
            ses::Vec3d{kTunnelK0, 0.0, 0.0},
            kTunnelDt,
        }};
    }

    double p_left_ = 0.0;
    double p_right_ = 0.0;
    double t_max_ = 0.0;
};

}  // namespace ses_shell
