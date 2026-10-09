// Landau2DDirector contracts (needs the GPU-layer library ses_scenario).

#include <gtest/gtest.h>

import ses.scenario.landau2d_director;

namespace {

// The central-difference ladder explodes near the lattice band top, so the
// scene guards each jump with a measurement-based rung check (the 1D ladder_cap
// rule): <H> must move by omega_c = B within 15%, else REFUSED.
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

}  // namespace
