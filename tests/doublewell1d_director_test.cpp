// DoubleWell1DDirector contracts (needs the GPU-layer library ses_scenario).

#include <gtest/gtest.h>

import ses.scenario.doublewell1d_director;

namespace {

// prepare_left must put psi in the LEFT well for every barrier the panel
// can set. psi_L = psi_0 + psi_1 needs psi_1 oriented positive on the left;
// on this box the odd state's sign is not fixed by the solver's u[0] rule
// (u[0] sits ~e^-270 deep in the forbidden wall), so it must be oriented
// explicitly.
TEST(DoubleWell1DDirector, PreparesTheLeftWellForEveryBarrier) {
    ses_shell::DoubleWell1DDirector d;
    ses_shell::DoubleWellApi* dw = d.doublewell();
    ASSERT_NE(dw, nullptr);
    EXPECT_GT(dw->p_left(), 0.9) << "boot barrier";
    constexpr int kSamples = 24;
    for (int s = 0; s <= kSamples; ++s) {
        const double vb = ses_shell::kDw1dBarrierMin +
                          (ses_shell::kDw1dBarrierMax - ses_shell::kDw1dBarrierMin) *
                              s / kSamples;
        dw->set_barrier(vb);
        EXPECT_GT(dw->p_left(), 0.9) << "Vb = " << vb;
    }
}

}  // namespace
