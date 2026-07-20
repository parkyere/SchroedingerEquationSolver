// RED: pick the resident eigenstate-atlas precision that fits GPU VRAM.
// Overflowing fp32 pages into WDDM shared RAM -> framerate collapse, so fall to fp16.
// Decision is pure integer arithmetic (this function); the VK_EXT_memory_budget
// query that feeds it is the untested Humble-Object shell (app/src/main.cpp).


#include <gtest/gtest.h>

#include <cstdint>

import ses.vram_budget;

namespace {

using ses::choose_state_precision;
using ses::GpuPrecision;

// Real workload: the n<=6 manifold at 256^3.
constexpr int kNumStates = 91;
constexpr std::int64_t kBytesPerStateFp32 = 134217728;  // 256^3 * 2 floats * 4 B
constexpr std::int64_t kHeadroom = 512LL * 1024 * 1024;  // textures/fbo/working
constexpr std::int64_t kGiB = 1024LL * 1024 * 1024;

TEST(ChooseStatePrecision, AmpleVramKeepsFp32) {
    bool fits = false;
    EXPECT_EQ(choose_state_precision(16 * kGiB, kNumStates, kBytesPerStateFp32,
                                     kHeadroom, &fits),
              GpuPrecision::Fp32);
    EXPECT_TRUE(fits);
}

TEST(ChooseStatePrecision, Fp32OverflowsButFp16Fits) {
    // 8 GB card: need32 ~= 12.2 GB > budget, need16 ~= 6.1 GB <= budget.
    bool fits = false;
    EXPECT_EQ(choose_state_precision(8 * kGiB, kNumStates, kBytesPerStateFp32,
                                     kHeadroom, &fits),
              GpuPrecision::Fp16);
    EXPECT_TRUE(fits);
}

TEST(ChooseStatePrecision, EvenFp16OverflowsStillPicksFp16AndFlags) {
    // 4 GB: even fp16 (~6.1 GB) overflows; pick smallest anyway, flag so caller can shrink the box.
    bool fits = true;
    EXPECT_EQ(choose_state_precision(4 * kGiB, kNumStates, kBytesPerStateFp32,
                                     kHeadroom, &fits),
              GpuPrecision::Fp16);
    EXPECT_FALSE(fits);
}

TEST(ChooseStatePrecision, UnmeasurableBudgetDefaultsToFp32) {
    // kVramUnknown = VK_EXT_memory_budget unavailable; do NOT downgrade fidelity on an unmeasurable budget.
    bool fits = false;
    EXPECT_EQ(choose_state_precision(ses::kVramUnknown, kNumStates,
                                     kBytesPerStateFp32, kHeadroom, &fits),
              GpuPrecision::Fp32);
    EXPECT_TRUE(fits);
    // Any negative (nonsense) free value: same default.
    EXPECT_EQ(choose_state_precision(-5, kNumStates, kBytesPerStateFp32,
                                     kHeadroom, nullptr),
              GpuPrecision::Fp32);
}

TEST(ChooseStatePrecision, OutFitsPointerIsOptional) {
    EXPECT_EQ(choose_state_precision(16 * kGiB, kNumStates, kBytesPerStateFp32,
                                     kHeadroom),
              GpuPrecision::Fp32);
}

}  // namespace
