// H2PlusDirector contracts on the CPU path (no engine is initialized here;
// needs the GPU-layer library ses_scenario).

#include <gtest/gtest.h>

#include <string>

import ses.scenario.molecule_director;

namespace {

// One director for the three label contracts: a 256^3 scene costs seconds.
//  - At boot the CPU path shows make()'s Gaussian seed: the title must not
//    claim an atlas orbital (the GPU path prepares 1sigma_g in on_gpu_ready).
//  - A prepared orbital is named.
//  - A position measurement collapses it into a packet: the name goes. (It
//    also decides what a device-loss fallback restores -- a kept index would
//    put the orbital back over the measured packet.)
TEST(H2PlusDirector, TheTitleNamesAnOrbitalOnlyWhileOneIsShown) {
    ses_shell::H2PlusDirector d;
    const auto names_an_orbital = [&d] {
        return d.title_text().find("showing") != std::string::npos;
    };
    EXPECT_FALSE(names_an_orbital()) << d.title_text();

    ses_shell::MoleculeApi* ml = d.molecule();
    ASSERT_NE(ml, nullptr);
    ASSERT_GT(ml->state_count(), 2);
    ml->prepare(2);
    EXPECT_TRUE(names_an_orbital()) << d.title_text();

    d.measure_now();
    EXPECT_FALSE(names_an_orbital()) << d.title_text();
}

}  // namespace
