module;
#include <algorithm>
export module ses.view_state;

// Shell view state behind the F / [ ] hotkeys and their panel widgets: flow
// tracer toggle + Beer-Lambert fog absorbance (clamped, multiplicative hotkey
// step). Pure value type; the shell and ImGui panel are glue over it.
// Contract: tests/view_state_test.cpp.

export namespace ses {

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
