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

}  // namespace
