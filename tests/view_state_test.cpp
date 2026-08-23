// Shell view state behind the F / [ ] hotkeys and their panel widgets: flow
// tracer toggle + Beer-Lambert fog absorbance with a clamped range and a
// multiplicative hotkey step. Pure value type; the shell (main.cpp) and the
// ImGui panel are thin glue over it. Contract for core/src/view_state.ixx.

#include <gtest/gtest.h>

import ses.view_state;

namespace {

using ses::ViewState;

TEST(ViewState, DefaultsMatchTheShell) {
    const ViewState v{};
    EXPECT_FALSE(v.flow);
    EXPECT_DOUBLE_EQ(v.absorbance, 0.68);
}

TEST(ViewState, ToggleFlowFlips) {
    ViewState v{};
    v.toggle_flow();
    EXPECT_TRUE(v.flow);
    v.toggle_flow();
    EXPECT_FALSE(v.flow);
}

TEST(ViewState, SetAbsorbanceClampsToRange) {
    ViewState v{};
    v.set_absorbance(0.0);
    EXPECT_DOUBLE_EQ(v.absorbance, ViewState::kAbsorbMin);
    v.set_absorbance(-3.0);
    EXPECT_DOUBLE_EQ(v.absorbance, ViewState::kAbsorbMin);
    v.set_absorbance(1e9);
    EXPECT_DOUBLE_EQ(v.absorbance, ViewState::kAbsorbMax);
    v.set_absorbance(2.5);
    EXPECT_DOUBLE_EQ(v.absorbance, 2.5);
}

TEST(ViewState, RangeIsTheHistoricalHotkeyRange) {
    EXPECT_DOUBLE_EQ(ViewState::kAbsorbMin, 0.1);
    EXPECT_DOUBLE_EQ(ViewState::kAbsorbMax, 50.0);
    EXPECT_DOUBLE_EQ(ViewState::kAbsorbStep, 1.3);
}

TEST(ViewState, StepAbsorbanceIsMultiplicativeAndClamped) {
    ViewState v{};
    v.set_absorbance(1.0);
    v.step_absorbance(+1);  // ']'
    EXPECT_DOUBLE_EQ(v.absorbance, 1.3);
    v.step_absorbance(-1);  // '['
    EXPECT_DOUBLE_EQ(v.absorbance, 1.0);
    // Stepping past either end saturates instead of escaping the range.
    v.set_absorbance(ViewState::kAbsorbMax);
    v.step_absorbance(+1);
    EXPECT_DOUBLE_EQ(v.absorbance, ViewState::kAbsorbMax);
    v.set_absorbance(ViewState::kAbsorbMin);
    v.step_absorbance(-1);
    EXPECT_DOUBLE_EQ(v.absorbance, ViewState::kAbsorbMin);
}

TEST(ViewState, StepZeroIsANoOp) {
    ViewState v{};
    v.set_absorbance(3.0);
    v.step_absorbance(0);
    EXPECT_DOUBLE_EQ(v.absorbance, 3.0);
}

// Streak advection step follows the pacing dial: x1 keeps the readable-crawl
// baseline, xN moves the tracers N times as far per frame.
TEST(FlowAdvectDt, BaselineCrawlAtRealTime) {
    EXPECT_DOUBLE_EQ(ses::flow_advect_dt(1), ses::kFlowCrawlDt);
}

TEST(FlowAdvectDt, ScalesLinearlyWithTheTimeScaleDial) {
    for (int s = 2; s <= 16; ++s) {
        EXPECT_DOUBLE_EQ(ses::flow_advect_dt(s),
                         static_cast<double>(s) * ses::kFlowCrawlDt);
    }
}

TEST(FlowAdvectDt, NonPositiveDialFallsBackToBaseline) {
    EXPECT_DOUBLE_EQ(ses::flow_advect_dt(0), ses::kFlowCrawlDt);
    EXPECT_DOUBLE_EQ(ses::flow_advect_dt(-3), ses::kFlowCrawlDt);
}

// Accumulation gate: any on-screen motion -- animating flow streaks OR
// in-flight overlay curves (photon streaks) -- must break the freeze.
TEST(AccumulateFrame, FreezesOnlyWhenTrulyStatic) {
    EXPECT_TRUE(ses::accumulate_frame(true, false, 0));
    EXPECT_FALSE(ses::accumulate_frame(false, false, 0));
    EXPECT_FALSE(ses::accumulate_frame(true, true, 0));
}

TEST(AccumulateFrame, InFlightOverlaysBreakTheFreeze) {
    EXPECT_FALSE(ses::accumulate_frame(true, false, 1));
    EXPECT_FALSE(ses::accumulate_frame(true, false, 16));
}

}  // namespace
