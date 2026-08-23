module;
#include <algorithm>
export module ses.view_state;

// Shell view state behind the F / [ ] hotkeys and their panel widgets: flow
// tracer toggle + Beer-Lambert fog absorbance (clamped, multiplicative hotkey
// step) + streakline advection step. Pure value type; the shell and ImGui
// panel are glue over it. Contract: tests/view_state_test.cpp.

export namespace ses {

// Streakline advection step per rendered frame (au): x1 readable-crawl
// baseline scaled by the pacing dial (tracer speed tracks sim speed).
inline constexpr double kFlowCrawlDt = 0.18;

inline constexpr double flow_advect_dt(int time_scale) noexcept {
    return kFlowCrawlDt * static_cast<double>(std::max(time_scale, 1));
}

// Temporal-accumulation gate: average only while NOTHING moves on screen --
// animating flow streaks AND in-flight overlay curves (photon streaks) must
// break the freeze, or their motion smears into the average.
inline constexpr bool accumulate_frame(bool scene_static, bool flow_animating,
                                       int overlay_curves) noexcept {
    return scene_static && !flow_animating && overlay_curves == 0;
}

struct ViewState {
    static constexpr double kAbsorbMin = 0.1;
    static constexpr double kAbsorbMax = 50.0;
    static constexpr double kAbsorbStep = 1.3;  // '[' divides, ']' multiplies

    bool flow = false;         // Key F: probability-current flow particles
    double absorbance = 0.68;  // Beer-Lambert opacity of the |psi|^2 fog

    void toggle_flow() noexcept { flow = !flow; }

    void set_absorbance(double a) noexcept {
        absorbance = std::clamp(a, kAbsorbMin, kAbsorbMax);
    }

    // dir > 0: one hotkey step denser (']'), dir < 0: one step thinner ('['),
    // 0: no-op. Saturates at the range ends.
    void step_absorbance(int dir) noexcept {
        if (dir > 0) {
            set_absorbance(absorbance * kAbsorbStep);
        } else if (dir < 0) {
            set_absorbance(absorbance / kAbsorbStep);
        }
    }
};

}  // namespace ses
