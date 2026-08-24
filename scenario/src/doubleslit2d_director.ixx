module;
#include <numbers>
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <volk.h>
export module ses.scenario.doubleslit2d_director;
export import ses.scenario;
export import ses.field;
export import ses.grid;
export import ses.lattice2d;
import ses.parallel;
import ses.potential;
import ses.vk.lattice2d_engine;


// 2D double-slit + Aharonov-Bohm: one normalized packet per shot into a
// two-slit wall; a z-solenoid hides between the slits; a screen line
// integrates arrivals across non-interacting shots.
// Peierls lattice (ses.lattice2d): flux enters as exact link phases,
// B = 0 in every reachable plaquette -- pure AB fringe shift, period 2 pi.
// Physics on one z-plane (nz = 1); staging replicates it into the display slab.
// volk.h textually first: VK_* macros never cross module boundaries.


export namespace ses_shell {

constexpr double kDs2dBoxX = 60.0;
constexpr double kDs2dBoxY = 60.0;
constexpr int kDs2dNx = 512;
constexpr int kDs2dNy = 512;
constexpr int kDs2dNz = 4;
constexpr double kDs2dZHalf = 2.0;
constexpr double kDs2dDt = 0.01;
// k0 = 1: fringe ~17 au at d = 8 (user's call).
constexpr double kDs2dK0 = 1.0;
constexpr double kDs2dSigma = 8.0;
constexpr double kDs2dLaunchX = -35.0;
constexpr double kDs2dWallLo = 0.0;
constexpr double kDs2dWallHi = 1.5;
constexpr double kDs2dWallV = 40.0;  // kappa*w ~ 13: leak e^-13, honest wall
constexpr double kDs2dSep = 8.0;
constexpr double kDs2dSepMin = 4.0;
constexpr double kDs2dSepMax = 16.0;
constexpr double kDs2dWidth = 2.0;
constexpr double kDs2dWidthMin = 1.0;
constexpr double kDs2dWidthMax = 4.0;
constexpr double kDs2dScreenX = 45.0;
constexpr double kDs2dAbsorb = 10.0;
// Quadratic CAP peak (ses::quadratic_cap_mask).
// CONTRACT: lattice2d_test SinglePacketDrainsThroughTheOpenBoundary.
constexpr double kDs2dAbsorbW0 = 4.0;
constexpr int kDs2dStepsPerTick = 16;  // ~9.6 au/s at 60 fps: ~9 s transit

class DoubleSlit2DDirector final : public ScenarioDirector, public SlitApi {
public:
    DoubleSlit2DDirector()
        : phys_grid_{ses::Grid1D{-kDs2dBoxX, kDs2dBoxX, kDs2dNx},
                     ses::Grid1D{-kDs2dBoxY, kDs2dBoxY, kDs2dNy},
                     ses::Grid1D{-1.0, 1.0, 1}},
          disp_grid_{ses::Grid1D{-kDs2dBoxX, kDs2dBoxX, kDs2dNx},
                     ses::Grid1D{-kDs2dBoxY, kDs2dBoxY, kDs2dNy},
                     ses::Grid1D{-kDs2dZHalf, kDs2dZHalf, kDs2dNz}},
          psi_{phys_grid_} {
        mask_ = ses::quadratic_cap_mask(phys_grid_, kDs2dAbsorb,
                                        kDs2dAbsorbW0, kDs2dDt);
        i_scr_ = ses::nearest_index(phys_grid_.x, kDs2dScreenX);
        rebuild_wall_and_prop();
        fire();
    }

    SlitApi* slit() override { return this; }

    // ---- SlitApi ----
    void set_separation(double d) override {
        sep_ = std::clamp(d, kDs2dSepMin, kDs2dSepMax);
        rebuild_wall_and_prop();
        fire();
    }
    double separation() const override { return sep_; }
    void set_width(double w) override {
        width_ = std::clamp(w, kDs2dWidthMin, kDs2dWidthMax);
        rebuild_wall_and_prop();
        fire();
    }
    double width() const override { return width_; }
    void set_flux(double phi) override {
        flux_ = phi;
        prop_->set_solenoid(flux_, solenoid_x(), 0.0);
        if (eng_ok_) {
            eng_.set_solenoid(flux_, solenoid_x(), 0.0);
        }
        fire();
        rebuild_props_overlays();  // arrow length encodes Phi
    }
    double flux() const override { return flux_; }
    // CONTRACT: the selftest arc's shot2 leg doubles the axis histogram.
    void refire() override { launch(); }
    double transmitted_fraction() const override {
        const double cell = phys_grid_.x.spacing() * phys_grid_.y.spacing() *
                            phys_grid_.z.spacing();
        double acc = 0.0;
        for (int j = 0; j < kDs2dNy; ++j) {
            for (int i = 0; i < kDs2dNx; ++i) {
                if (phys_grid_.x.coord(i) > kDs2dWallHi) {
                    acc += std::norm(psi_(i, j, 0));
                }
            }
        }
        return acc * cell;
    }
    double screen_at(double y) const override {
        return screen_[static_cast<std::size_t>(
            ses::nearest_index(phys_grid_.y, y))];
    }
    double k0() const override { return kDs2dK0; }
    double screen_distance() const override {
        return kDs2dScreenX - 0.5 * (kDs2dWallLo + kDs2dWallHi);
    }

    // ---- lifecycle ----
    const ses::Grid3D& grid() const override { return disp_grid_; }
    // Peierls + solenoid scene: GPU port is ses_vk::Lattice2DEngine (exact link
    // phases, vkcheck-certified). The edge CAP runs on the GPU too (no renorm:
    // the surviving norm IS the transmitted fraction), so each step only reads
    // back for the per-step screen integral -- no re-upload.
    void init_compute(ses_vk::DeviceContext& ctx, bool device_ok,
                      std::int64_t /*free_vram*/) override {
        compute_attempted_ = true;
        if (device_ok && eng_.initialize(ctx)) {
            eng_.set_lattice(phys_grid_, wallv_, kDs2dDt);
            eng_.set_solenoid(flux_, solenoid_x(), 0.0);
            eng_.set_absorber(mask_);
            eng_.upload(psi_.data());
            eng_ok_ = true;
            eng_dirty_ = false;
        }
    }
    void release_gpu() override {
        eng_.destroy();
        eng_ok_ = false;
        eng_dirty_ = true;
    }
    bool compute_attempted() const override { return compute_attempted_; }
    bool gpu_ok() const override { return false; }

    // ---- per-frame ----
    void run_frame() override {
        if (pending_steps_ > 0) {
            const int n = pending_steps_;
            pending_steps_ = 0;
            step_batch(n);
            sim_time_ += n * kDs2dDt;
            dirt_.mark_display();
            dirt_.staging_dirty = true;
        }
        if (dirt_.staging_dirty) {
            dirt_.staging_dirty = false;
            pack_plane_staging(psi_.data(), kDs2dNz, staging_);
            rebuild_screen_curve();
        }
        if (++frames_ % 10 == 0) {
            dirt_.title_dirty = true;
        }
    }
    void tick() override {
        const int per_tick = kDs2dStepsPerTick * time_scale_;
        pending_steps_ = pending_after_tick(pending_steps_, per_tick);
    }

    // ---- controls ----
    void do_set_real_time() override {}
    void reset_simulation() override { fire(); }
    void measure_now() override {}
    void toggle_view_mode() override {}  // cloud-only scene
    bool handle_key(char key) override {
        if (key == '2') {
            fire();
            return true;
        }
        return false;
    }

    bool solving() const override { return false; }
    bool scene_ready() const override { return true; }
    void set_time_scale(int scale) override {
        time_scale_ = clamp_time_scale(scale);
    }
    int time_scale() const override { return time_scale_; }
    double sim_time() const override { return sim_time_; }
    double sim_dt() const override { return kDs2dDt; }
    int steps_per_tick_x1() const override { return kDs2dStepsPerTick; }

    // ---- display ----
    bool cloud() const override { return true; }
    double peak() const override { return peak_; }
    VkImageView psi_volume_view() override { return VK_NULL_HANDLE; }
    float next_flash_intensity() override { return 0.0f; }
    bool take_volume_written() override {
        return dirt_.take_volume_written();
    }
    bool take_volume_dirty() override { return dirt_.take_volume_dirty(); }
    bool take_mesh_dirty() override { return false; }
    void mark_display_dirty() override { dirt_.mark_display(); }
    bool take_title_dirty() override { return dirt_.take_title_dirty(); }
    const std::vector<float>& psi_staging() const override {
        return staging_;
    }
    const ses::Mesh& mesh() const override { return no_mesh_; }
    const std::vector<ses::Rgb>& colors() const override {
        return no_colors_;
    }
    std::string title_text() override {
        const double pi = std::numbers::pi;
        return strf(
            "Electron double slit + Aharonov-Bohm (2D lattice)  |  t = {:.1f} "
            "au ({}x{}, dt {:.2g})  d = {:.1f}  w = {:.1f}  Phi = {:.2f} pi  "
            "shot {}  T = {:.1f}%  B = 0 on every electron path  "
            "keys: 2 fire electron",
            sim_time_, kDs2dNx, kDs2dNy, kDs2dDt, sep_, width_, flux_ / pi,
            shots_, 100.0 * transmitted_fraction());
    }

    int marker_count() const override { return 0; }

    int overlay_curve_count() const override { return 4; }
    OverlayCurve overlay_curve(int i) const override {
        struct Desc {
            const std::vector<float>* pts;
            float r, g, b, a;
            bool fill;
        };
        const Desc table[4] = {
            {&wall_curve_, 1.0f, 0.30f, 0.25f, 0.55f, true},
            {&solenoid_curve_, 1.0f, 0.70f, 0.20f, 1.0f, false},
            {&plate_curve_, 0.85f, 0.85f, 0.90f, 0.35f, false},
            {&screen_curve_, 0.35f, 0.85f, 1.0f, 0.95f, false},
        };
        const Desc& d = table[std::clamp(i, 0, 3)];
        return {d.pts->data(), static_cast<int>(d.pts->size() / 3),
                d.r, d.g, d.b, d.a, d.fill};
    }

    double default_camera_azimuth() const override { return 0.0; }
    double default_camera_elevation() const override { return 0.0; }
    double default_camera_distance() const override { return 170.0; }

private:
    double solenoid_x() const {
        // Quarter-cell off a lattice line: unambiguous host plaquette.
        return 0.5 * (kDs2dWallLo + kDs2dWallHi) +
               0.25 * phys_grid_.x.spacing();
    }

    bool slit_open(double y) const {
        return std::abs(y - 0.5 * sep_) <= 0.5 * width_ ||
               std::abs(y + 0.5 * sep_) <= 0.5 * width_;
    }

    void rebuild_wall_and_prop() {
        std::vector<double> v(static_cast<std::size_t>(phys_grid_.size()),
                              0.0);
        for (int j = 0; j < kDs2dNy; ++j) {
            const double y = phys_grid_.y.coord(j);
            if (slit_open(y)) {
                continue;
            }
            for (int i = 0; i < kDs2dNx; ++i) {
                const double x = phys_grid_.x.coord(i);
                if (x >= kDs2dWallLo && x <= kDs2dWallHi) {
                    v[static_cast<std::size_t>(phys_grid_.flat(i, j, 0))] =
                        kDs2dWallV;
                }
            }
        }
        prop_ = std::make_unique<ses::PeierlsLattice2D>(phys_grid_, v,
                                                        kDs2dDt);
        prop_->set_solenoid(flux_, solenoid_x(), 0.0);
        wallv_ = std::move(v);  // kept for the GPU engine
        if (eng_ok_) {
            eng_.set_lattice(phys_grid_, wallv_, kDs2dDt);
            eng_.set_solenoid(flux_, solenoid_x(), 0.0);
            eng_.set_absorber(mask_);  // set_lattice resets links/absorber
        }
        rebuild_props_overlays();
    }

    // One normalized packet replaces the in-flight one; screen survives (vs fire()).
    void launch() {
        ses::parallel_for(kDs2dNy, [&](int j) {
            const double y = phys_grid_.y.coord(j);
            for (int i = 0; i < kDs2dNx; ++i) {
                const double x = phys_grid_.x.coord(i);
                const double dx = x - kDs2dLaunchX;
                psi_(i, j, 0) =
                    std::exp(-(dx * dx + y * y) /
                             (4.0 * kDs2dSigma * kDs2dSigma)) *
                    std::polar(1.0, kDs2dK0 * x);
            }
        });
        const double n = ses::norm_sq(psi_);
        if (n > 0.0) {
            const double inv = 1.0 / std::sqrt(n);
            for (auto& c : psi_.data()) {
                c *= inv;
            }
        }
        ++shots_;
        peak_ = 1.0;
        eng_dirty_ = true;  // new packet on the CPU: re-upload before stepping
        dirt_.mark_all_dirty();
    }

    // Full reset; geometry/flux edits route here.
    void fire() {
        screen_.assign(static_cast<std::size_t>(kDs2dNy), 0.0);
        sim_time_ = 0.0;
        pending_steps_ = 0;
        shots_ = 0;
        launch();
    }

    void step_batch(int n) {
        if (eng_ok_ && eng_dirty_) {
            eng_.upload(psi_.data());
            eng_dirty_ = false;
        }
        for (int s = 0; s < n; ++s) {
            if (eng_ok_) {
                // GPU step + edge CAP (no renorm: norm = electron on stage).
                // Read back every step for the screen integral; the CAP is on
                // the GPU so the state stays in sync -- no re-upload.
                eng_.step(1);
                eng_.download();
                unpack_interleaved(eng_.state(), psi_.data());
            } else {
                prop_->step(psi_);
                ses::parallel_for(kDs2dNy, [&](int j) {
                    const std::size_t row = static_cast<std::size_t>(j) *
                                            static_cast<std::size_t>(kDs2dNx);
                    for (int i = 0; i < kDs2dNx; ++i) {
                        const std::size_t c = row + static_cast<std::size_t>(i);
                        psi_.data()[c] *=
                            mask_[static_cast<std::size_t>(j * kDs2dNx + i)];
                    }
                });
            }
            for (int j = 0; j < kDs2dNy; ++j) {
                screen_[static_cast<std::size_t>(j)] +=
                    std::norm(psi_(i_scr_, j, 0)) * kDs2dDt;
            }
        }
        double pk = 0.0;
        for (int j = 0; j < kDs2dNy; ++j) {
            for (int i = 0; i < kDs2dNx; ++i) {
                pk = std::max(pk, std::norm(psi_(i, j, 0)));
            }
        }
        peak_ = std::max(pk, kPeakDecay * peak_);
    }

    void rebuild_props_overlays() {
        wall_curve_.clear();
        auto slab = [&](double y0, double y1) {
            if (y1 > y0) {  // middle slab can vanish at wide slits
                strip_wall_quad(wall_curve_, kDs2dWallLo, kDs2dWallHi, y0,
                                y1);
            }
        };
        slab(-kDs2dBoxY, -0.5 * sep_ - 0.5 * width_);
        slab(-0.5 * sep_ + 0.5 * width_, 0.5 * sep_ - 0.5 * width_);
        slab(0.5 * sep_ + 0.5 * width_, kDs2dBoxY);

        // Solenoid: core circle + z arrow, signed length ~ Phi.
        solenoid_curve_.clear();
        const double pi = std::numbers::pi;
        const float xs = static_cast<float>(solenoid_x());
        auto put = [&](float x, float y, float z) {
            solenoid_curve_.push_back(x);
            solenoid_curve_.push_back(y);
            solenoid_curve_.push_back(z);
        };
        const float rc = 1.2f;
        for (int t = 0; t <= 24; ++t) {
            const double th = 2.0 * pi * t / 24.0;
            put(xs + rc * static_cast<float>(std::cos(th)),
                rc * static_cast<float>(std::sin(th)), 0.0f);
        }
        const float len =
            static_cast<float>(5.0 * flux_ / (2.0 * pi));
        put(xs, 0.0f, 0.0f);
        put(xs, 0.0f, len);
        const float back = len - 0.18f * len - std::copysign(0.6f, len);
        put(xs - 0.8f, 0.0f, back);
        put(xs, 0.0f, len);
        put(xs + 0.8f, 0.0f, back);

        plate_curve_ = {static_cast<float>(kDs2dScreenX),
                        -static_cast<float>(kDs2dBoxY - kDs2dAbsorb), 0.0f,
                        static_cast<float>(kDs2dScreenX),
                        static_cast<float>(kDs2dBoxY - kDs2dAbsorb), 0.0f};
    }

    void rebuild_screen_curve() {
        double smax = 0.0;
        for (const double s : screen_) {
            smax = std::max(smax, s);
        }
        // 1e-6 floor: else the self-normalized tail (~1e-22) renders as a fake pattern.
        const double scale = smax > 1e-6 ? 12.0 / smax : 0.0;
        screen_curve_.clear();
        for (int j = 0; j < kDs2dNy; ++j) {
            const double y = phys_grid_.y.coord(j);
            if (std::abs(y) > kDs2dBoxY - kDs2dAbsorb) {
                continue;
            }
            screen_curve_.push_back(static_cast<float>(
                kDs2dScreenX + scale * screen_[static_cast<std::size_t>(j)]));
            screen_curve_.push_back(static_cast<float>(y));
            screen_curve_.push_back(0.0f);
        }
    }

    ses::Grid3D phys_grid_;
    ses::Grid3D disp_grid_;
    ses::Field3D psi_;
    std::unique_ptr<ses::PeierlsLattice2D> prop_;  // CPU fallback
    ses_vk::Lattice2DEngine eng_;                  // GPU Peierls propagator
    bool eng_ok_ = false;
    bool eng_dirty_ = true;
    std::vector<double> wallv_;                    // wall potential (for engine)
    std::vector<double> mask_;
    std::vector<double> screen_;
    std::vector<float> staging_;
    std::vector<float> wall_curve_;
    std::vector<float> solenoid_curve_;
    std::vector<float> plate_curve_;
    std::vector<float> screen_curve_;
    double sep_ = kDs2dSep;
    double width_ = kDs2dWidth;
    double flux_ = 0.0;
    double peak_ = 1.0;
    double sim_time_ = 0.0;
    int pending_steps_ = 0;
    int time_scale_ = 1;
    int i_scr_ = 0;  // screen column, fixed by the grid (set once in the ctor)
    std::uint64_t frames_ = 0;
    DisplayFlags dirt_;
    bool compute_attempted_ = false;

    ses::Mesh no_mesh_;
    int shots_ = 0;
    std::vector<ses::Rgb> no_colors_;
};

}  // namespace ses_shell
