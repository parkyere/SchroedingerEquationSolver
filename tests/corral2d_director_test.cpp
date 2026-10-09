// Corral2DDirector contracts (needs the GPU-layer library ses_scenario).

#include <gtest/gtest.h>

import ses.scenario.corral2d_director;

namespace {

// fermi_wave() = standing wave at E_F (k_F R = j0_10, ~10 nodes): the 1993
// topograph images E_F LDOS, NOT the ground.
TEST(Corral2DDirector, FermiWaveIsQuasiStationary) {
    ses_shell::Corral2DDirector d;
    ses_shell::CorralApi* api = d.corral();
    ASSERT_NE(api, nullptr);
    api->fermi_wave();
    const double conf0 = api->confinement();
    EXPECT_GT(conf0, 0.75);  // J0 tail reaches past R by construction
    for (int t = 0; t < 30; ++t) {
        d.tick();
        d.run_frame();
    }
    // measured ~86% over ~5 au; the fence absorbs by design, floor 0.8.
    EXPECT_GT(api->confinement(), 0.8 * conf0);
}

}  // namespace
