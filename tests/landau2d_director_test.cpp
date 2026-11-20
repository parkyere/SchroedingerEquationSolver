// Landau2DDirector contracts (needs the GPU-layer library ses_scenario).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

import ses.scenario.landau2d_director;

namespace {

// Largest |psi|^2 in the staged display volume (interleaved re, im floats).
double staged_peak(const std::vector<float>& s) {
    double pk = 0.0;
    for (std::size_t i = 0; i + 1 < s.size(); i += 2) {
        const double re = s[i];
        const double im = s[i + 1];
        pk = std::max(pk, re * re + im * im);
    }
    return pk;
}

// The central-difference ladder explodes near the lattice band top, so the
// scene refuses an up-jump once <H> would exceed the band top 0.30/h^2 and a
// down-jump that removes no clean quantum (see Landau2DDirector::ladder).
TEST(Landau2DDirector, LadderRefusesPastTheLatticeBand) {
    ses_shell::Landau2DDirector d;
    ses_shell::LandauApi* api = d.landau();
    ASSERT_NE(api, nullptr);
    int climbed = 0;
    while (climbed < 40 && api->ladder(true)) {
        ++climbed;
    }
    // The band ceiling scales as h^-2, so the rung count tracks the grid:
    // never immediately, never unbounded.
    EXPECT_GE(climbed, 15);
    EXPECT_LE(climbed, 35);
    // the Landau index followed the rungs
    const double top_n = api->mean_n();
    EXPECT_GT(top_n, 0.6 * climbed);
    // Descending is refused on these coherent-displaced states: a|alpha> ~
    // alpha|alpha> removes no clean quantum, the energy drop is ~0, the floor
    // guard strikes at once. Contract = TERMINATION + monotonicity, not unwinding.
    int descended = 0;
    while (descended < 40 && api->ladder(false)) {
        ++descended;
    }
    EXPECT_LT(descended, 40);
    EXPECT_LE(api->mean_n(), top_n + 1e-9);
}

// peak() normalizes the cloud's brightness, so it must follow the state: a
// ladder jump replaces psi and the normalizer may not stay at the launch value.
TEST(Landau2DDirector, DisplayPeakFollowsALadderJump) {
    ses_shell::Landau2DDirector d;
    ses_shell::LandauApi* api = d.landau();
    ASSERT_NE(api, nullptr);
    d.run_frame();  // packs the display plane
    const double launch = staged_peak(d.psi_staging());
    ASSERT_GT(launch, 0.0);
    EXPECT_NEAR(d.peak(), launch, 1e-5 * launch);
    for (int k = 0; k < 4; ++k) {
        ASSERT_TRUE(api->ladder(true)) << "rung " << k;
    }
    d.run_frame();
    const double now = staged_peak(d.psi_staging());
    ASSERT_GT(std::abs(now - launch), 1e-2 * launch)
        << "the jumps must change the peak for this test to mean anything";
    EXPECT_NEAR(d.peak(), now, 1e-5 * now);
}

}  // namespace
