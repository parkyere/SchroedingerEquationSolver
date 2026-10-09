// Harmonic1DDirector contracts (needs the GPU-layer library ses_scenario).

#include <gtest/gtest.h>

import ses.scenario.harmonic1d_director;

namespace {

// kappa > 0 no-jump damping mutates psi every step; the strip must track it
// BETWEEN jumps, not only on parity flips (stale-cache regression).
TEST(HoSpectrum, StripTracksMcwfDampingBetweenJumps) {
    ses_shell::Harmonic1DDirector d;
    ses_shell::Ladder1dApi* ld = d.ladder1d();
    ASSERT_NE(ld, nullptr);
    ld->cat();
    ld->toggle_loss();
    // Prime the lazy cache the way the app does (strip drawn every frame):
    // without a prior read the first read after stepping is trivially fresh
    // and the stale path is invisible.
    ASSERT_GT(ld->spectrum_count(), 0);
    // Land right after >= 25 consecutive no-jump steps: the stale path would
    // still show the spectrum frozen at the last flip (or at t = 0).
    int quiet = 0;
    int steps = 0;
    long long jumps = ld->jump_count();
    while (quiet < 25 && steps < 3000) {
        d.tick();
        d.run_frame();  // steps_per_tick = 1 at scale 1
        ++steps;
        const long long j = ld->jump_count();
        quiet = j == jumps ? quiet + 1 : 0;
        jumps = j;
    }
    ASSERT_GE(quiet, 25);
    const double live = ld->level_energy();  // Ha, moved by the damping
    double tot = 0.0;
    double mean_ev = 0.0;
    const int n = ld->spectrum_count();
    for (int i = 0; i < n; ++i) {
        tot += ld->spectrum_weight(i);
        mean_ev += ld->spectrum_ev(i) * ld->spectrum_weight(i);
    }
    ASSERT_GT(tot, 0.9);
    EXPECT_NEAR(mean_ev / tot / ses_shell::kHaToEv, live, 5e-3);
}

}  // namespace
