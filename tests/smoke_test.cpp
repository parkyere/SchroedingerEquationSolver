#include <gtest/gtest.h>

// Harness smoke test.
//
// Proves that the GoogleTest harness compiles, links against sesolver_core,
// and runs under ctest. It asserts nothing about the project itself.
//
// Real behavior tests are added test-first as features are built
// (see docs/TDD_RULES.md and docs/ROADMAP.md). Delete this file once the
// first genuine test exists.
TEST(Harness, SanityArithmetic) {
    EXPECT_EQ(2 + 2, 4);
}
