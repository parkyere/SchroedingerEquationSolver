// Rutherford3DDirector / BaseDirector contracts on the CPU path (no engine is
// initialized here; needs the GPU-layer library ses_scenario).

#include <gtest/gtest.h>

import ses.scenario.rutherford3d_director;

namespace {

// One director for both contracts: a 256^3 scene costs seconds to build.
//  - When the CPU path restages the cloud the shell must hear that the picture
//    changed: take_volume_written() resets the temporal accumulation and
//    rebuilds the occupancy/shadow volumes built from the old field.
//  - The panel button reads "Fire alpha (2)": key 2 must re-fire. The collapse
//    makes the re-fire observable through the display peak.
TEST(Rutherford3DDirector, CpuRestageIsReportedAndKeyTwoRefires) {
    ses_shell::Rutherford3DDirector d;
    const double launch = d.peak();
    (void)d.take_volume_dirty();
    (void)d.take_volume_written();
    d.measure_now();  // a narrower packet somewhere else
    ASSERT_NE(d.peak(), launch);
    EXPECT_TRUE(d.take_volume_dirty());
    EXPECT_TRUE(d.take_volume_written());
    EXPECT_TRUE(d.handle_key('2'));
    EXPECT_DOUBLE_EQ(d.peak(), launch);
}

}  // namespace
