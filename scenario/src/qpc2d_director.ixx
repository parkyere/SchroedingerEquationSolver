module;
#include <numbers>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.qpc2d_director;
export import ses.scenario.lattice2d_director;
import ses.potential;
import ses.propagator;
import ses.field;
import ses.parallel;
import ses.vk.engine_blobs;


// Quantum point contact: transmission staircase vs gap width.
// CONTRACT: tests/qpc2d_test.cpp + --selftest-qpc2d.


export namespace ses_shell {

// half-cosine lip on gap edges (raw corners -> Gibbs fringes).
inline std::vector<double> qpc_potential(const ses::Grid3D& g, double w,
                                         double wall_lo, double wall_hi,
                                         double v0, double lip) {
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const double pi = std::numbers::pi;
    for (int j = 0; j < g.y.n; ++j) {
        const double ay = std::abs(g.y.coord(j));
        double f = 1.0;
        if (ay < 0.5 * w) {
            f = 0.0;
        } else if (ay < 0.5 * w + lip) {
            const double t = (ay - 0.5 * w) / lip;
            const double s = std::sin(0.5 * pi * t);
            f = s * s;
        }
        if (f <= 0.0) {
            continue;
        }
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            if (x >= wall_lo && x <= wall_hi) {
                v[static_cast<std::size_t>(g.flat(i, j, 0))] = v0 * f;
            }
        }
    }
    return v;
}

constexpr double kQp2dBox = 30.0;
constexpr int kQp2dN = 512;
constexpr int kQp2dNz = 4;
constexpr double kQp2dZHalf = 2.0;
constexpr double kQp2dDt = 0.01;
constexpr double kQp2dK0 = 1.0;      // lambda/2 = pi: mode thresholds
constexpr double kQp2dWallLo = 0.0;
constexpr double kQp2dWallHi = 1.5;
constexpr double kQp2dWallV = 40.0;
constexpr double kQp2dLip = 0.8;
constexpr double kQp2dGap = 4.5;     // boot: one channel open
constexpr double kQp2dGapMin = 1.0;
constexpr double kQp2dGapMax = 10.0;
constexpr double kQp2dLaunchX = -18.0;
constexpr double kQp2dSigX = 4.0;
constexpr double kQp2dSigY = 10.0;   // WIDE front: many-mode illumination
constexpr double kQp2dCapW = 5.0;
constexpr double kQp2dCapW0 = 4.0;
constexpr int kQp2dStepsPerTick = 16;
constexpr double kQp2dSurfH = 6.0;
constexpr int kQp2dMeshStride = 1;

class Qpc2DDirector final : public Lattice2DDirectorBase, public QpcApi {
public:
    Qpc2DDirector()
        : Lattice2DDirectorBase(kQp2dBox, kQp2dN, kQp2dNz, kQp2dZHalf) {
        cap_ = ses::quadratic_cap_mask(phys_grid_, kQp2dCapW, kQp2dCapW0,
                                       kQp2dDt);
        rebuild_prop();
        fire();
    }

    // B = 0 spectral scene: GPU is the split-operator Engine at nz = 1 (VkFFT
    // 2D), certified by vkcheck check_engine_planar. The right-edge CAP tally
    // (transmitted fraction) is a delicate contract, so the CAP + tally stay on
    // the CPU exactly as before; only the propagation moves to the GPU.
    void init_compute(ses_vk::DeviceContext& ctx, bool device_ok,
                      std::int64_t /*free_vram*/) override {
        compute_attempted_ = true;
        eng_ok_ = device_ok &&
                  eng_.initialize(ctx, phys_grid_, ses_vk::engine_blobs(kQp2dN),
                                  v_, kQp2dDt, psi_.data());
        eng_dirty_ = false;
    }
    void release_gpu() override {
        eng_.destroy();
        eng_ok_ = false;
        eng_dirty_ = true;
    }

    QpcApi* qpc() override { return this; }

    // ---- QpcApi ----
    void set_gap(double w) override {
        gap_ = std::clamp(w, kQp2dGapMin, kQp2dGapMax);
        rebuild_prop();
        fire();
    }
    double gap() const override { return gap_; }
    void fire() override {
        ses::parallel_for(kQp2dN, [&](int j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kQp2dN; ++i) {
                const double x = phys_grid_.x.coord(i);
                const double dx = x - kQp2dLaunchX;
                psi_(i, j, 0) =
                    std::exp(-dx * dx / (4.0 * kQp2dSigX * kQp2dSigX) -
                             y * y / (4.0 * kQp2dSigY * kQp2dSigY)) *
                    std::polar(1.0, kQp2dK0 * x);
            }
        });
        ses::normalize(psi_);
        transmitted_ = 0.0;
        disp_peak_ = 0.0;
        eng_dirty_ = true;  // psi_ edited on the CPU: re-upload before stepping
        track_peak();
        mark_fired();
    }
    double transmitted() const override { return transmitted_; }
    int open_channels() const override {
        return static_cast<int>(gap_ * kQp2dK0 / std::numbers::pi);
    }

    void reset_simulation() override { fire(); }

    bool handle_key(char key) override {
        if (key == '2') {
            fire();
            return true;
        }
        return false;
    }

    double sim_dt() const override { return kQp2dDt; }
    int steps_per_tick() const override { return kQp2dStepsPerTick; }

    // ---- STM-style surface display ----
    bool cloud() const override { return false; }
    double default_camera_azimuth() const override { return 0.35; }
    double default_camera_elevation() const override { return 0.95; }
    double default_camera_distance() const override { return 110.0; }

    int overlay_curve_count() const override { return 1; }
    OverlayCurve overlay_curve(int /*i*/) const override {
        return {wall_curve_.data(),
                static_cast<int>(wall_curve_.size() / 3),
                1.0f, 0.30f, 0.25f, 0.55f, true};
    }

    std::string title_text() override {
        return strf(
            "Quantum point contact  |  t = {:.1f} au  gap = {:.1f} "
            "(lambda/2 = {:.2f}, channels {})  transmitted {:.1f}%  "
            "keys: 2 fire",
            sim_time_, gap_, std::numbers::pi / kQp2dK0,
            open_channels(), 100.0 * transmitted_);
    }

protected:
    void rebuild_display() override {
        rebuild_surface(psi_, kQp2dSurfH, kQp2dMeshStride);
    }

    // right-of-wall CAP absorption tallied as transmitted fraction. The GPU
    // does the propagation; the CAP edits psi_ on the CPU each step (contract
    // parity), so re-upload before the next GPU step.
    void do_steps(int n) override {
        if (eng_ok_) {
            if (eng_dirty_) {
                eng_.upload_state(psi_.data());
                eng_dirty_ = false;
            }
            for (int s = 0; s < n; ++s) {
                eng_.step(1);
                eng_.readback(rb_);
                unpack_interleaved(rb_.data(), psi_.data());
                apply_cap_and_tally();
                eng_.upload_state(psi_.data());  // psi_ damped: re-sync
            }
        } else {
            for (int s = 0; s < n; ++s) {
                prop_->step(psi_, 1);
                apply_cap_and_tally();
            }
        }
        sim_time_ += n * kQp2dDt;
        track_peak();
    }

private:
    void apply_cap_and_tally() {
        const double cell = phys_grid_.x.spacing() * phys_grid_.y.spacing() *
                            phys_grid_.z.spacing();
        double gained = 0.0;
        for (int j = 0; j < kQp2dN; ++j) {
            const std::size_t row = static_cast<std::size_t>(j) * kQp2dN;
            for (int i = 0; i < kQp2dN; ++i) {
                const std::size_t c = row + static_cast<std::size_t>(i);
                const double m = cap_[c];
                if (m < 1.0 && phys_grid_.x.coord(i) > kQp2dWallHi) {
                    gained += std::norm(psi_.data()[c]) * (1.0 - m * m) * cell;
                }
                psi_.data()[c] *= m;
            }
        }
        transmitted_ += gained;
    }

    void rebuild_prop() {
        v_ = qpc_potential(phys_grid_, gap_, kQp2dWallLo, kQp2dWallHi,
                           kQp2dWallV, kQp2dLip);
        prop_ = std::make_unique<ses::SplitOperator3D>(phys_grid_, v_,
                                                       kQp2dDt);
        if (eng_ok_) {
            eng_.set_potential(v_);  // gap changed: refresh the GPU potential
        }
        wall_curve_.clear();
        strip_wall_quad(wall_curve_, kQp2dWallLo, kQp2dWallHi, -kQp2dBox,
                        -0.5 * gap_);
        strip_wall_quad(wall_curve_, kQp2dWallLo, kQp2dWallHi, 0.5 * gap_,
                        kQp2dBox);
    }

    std::vector<double> v_;
    std::vector<double> cap_;
    std::unique_ptr<ses::SplitOperator3D> prop_;  // CPU fallback
    ses_vk::Engine eng_;                          // GPU: planar split-operator
    bool eng_ok_ = false;
    bool eng_dirty_ = true;
    std::vector<float> rb_;                        // readback scratch
    std::vector<float> wall_curve_;
    double gap_ = kQp2dGap;
    double transmitted_ = 0.0;
};

}  // namespace ses_shell
