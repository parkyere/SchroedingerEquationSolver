// RED: set_real_time() must restore time_scale()==1 on every director family.
// Root cause: set_real_time never touched time_scale_ (1D/2D overrides no-ops,
// base only flips stepping_) -> slider x16 stayed sticky.

#include <gtest/gtest.h>

import ses.scenario.tunneling1d_director;
import ses.scenario.doubleslit2d_director;

namespace {

// 1D family: set_real_time was a pure no-op here (override did nothing).
TEST(TimeScaleRealTime, Line1dFamilyRestoresUnitScale) {
    ses_shell::Tunneling1DDirector d;
    d.set_time_scale(16);
    ASSERT_EQ(d.time_scale(), 16);
    d.set_real_time();
    EXPECT_EQ(d.time_scale(), 1);
}

// Leaf-local time_scale_ copy (derives ScenarioDirector directly).
TEST(TimeScaleRealTime, Lattice2dFamilyRestoresUnitScale) {
    ses_shell::DoubleSlit2DDirector d;
    d.set_time_scale(16);
    ASSERT_EQ(d.time_scale(), 16);
    d.set_real_time();
    EXPECT_EQ(d.time_scale(), 1);
}

// The shared pacing invariants themselves (formerly copy-pasted per director).

TEST(TickPacing, DialClampsToTheHistoricalRange) {
    EXPECT_EQ(ses_shell::clamp_time_scale(0), 1);
    EXPECT_EQ(ses_shell::clamp_time_scale(-5), 1);
    EXPECT_EQ(ses_shell::clamp_time_scale(1), 1);
    EXPECT_EQ(ses_shell::clamp_time_scale(16), 16);
    EXPECT_EQ(ses_shell::clamp_time_scale(99), 16);
}

TEST(TickPacing, CatchUpTicksDropInsteadOfBundling) {
    EXPECT_EQ(ses_shell::pending_after_tick(0, 16), 16);
    // Un-consumed pending from a stalled frame must NOT stack up.
    EXPECT_EQ(ses_shell::pending_after_tick(16, 16), 16);
    EXPECT_EQ(ses_shell::pending_after_tick(3, 8), 8);
    EXPECT_EQ(ses_shell::pending_after_tick(0, 1), 1);
}

}  // namespace
