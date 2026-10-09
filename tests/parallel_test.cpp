// ses.parallel: worker pool replacing OpenMP, whose pragmas MSVC miscompiles
// inside C++20 module interfaces (see core/src/parallel.ixx).
// parallel_sum combines partials in fixed chunk order -> bitwise-identical
// run-to-run and for any worker count (OpenMP reduction never promised this;
// CPU is the oracle, so determinism outranks speed). parallel_ranges' worker
// index (0 <= worker < parallel_workers()) selects per-worker scratch, the
// thread_local replacement.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

import ses.parallel;

namespace {

TEST(ParallelWorkers, AtLeastOne) {
    EXPECT_GE(ses::parallel_workers(), 1);
}

TEST(ParallelFor, CoversEveryIndexExactlyOnce) {
    for (const int n : {0, 1, 3, 64, 10007}) {
        std::vector<int> hits(static_cast<std::size_t>(n), 0);
        ses::parallel_for(n, [&](int i) { ++hits[static_cast<std::size_t>(i)]; });
        for (int i = 0; i < n; ++i) {
            ASSERT_EQ(hits[static_cast<std::size_t>(i)], 1) << "n=" << n << " i=" << i;
        }
    }
}

TEST(ParallelFor, NestedCallDoesNotDeadlockAndStaysCorrect) {
    const int outer = 4;
    const int inner = 1000;
    std::vector<std::int64_t> sums(outer, 0);
    ses::parallel_for(outer, [&](int o) {
        std::int64_t s = 0;
        std::vector<int> hits(inner, 0);
        ses::parallel_for(inner, [&](int i) { ++hits[static_cast<std::size_t>(i)]; });
        for (int i = 0; i < inner; ++i) {
            s += hits[static_cast<std::size_t>(i)] * (i + 1);
        }
        sums[static_cast<std::size_t>(o)] = s;
    });
    const std::int64_t expect = static_cast<std::int64_t>(inner) * (inner + 1) / 2;
    for (int o = 0; o < outer; ++o) {
        EXPECT_EQ(sums[static_cast<std::size_t>(o)], expect);
    }
}

TEST(ParallelFor, BodyExceptionPropagatesAndPoolStaysParallel) {
    // A throwing body must surface on the caller, after EVERY worker has
    // left the region (the Job lives on the caller's stack), and must not
    // leave the pool believing a region is still open (which would run
    // every later region serially on the caller).
    EXPECT_THROW(ses::parallel_for(4096, [](int) { throw std::runtime_error("boom"); }),
                 std::runtime_error);

    std::vector<int> hits(10007, 0);
    ses::parallel_for(10007, [&](int i) { ++hits[static_cast<std::size_t>(i)]; });
    for (int i = 0; i < 10007; ++i) {
        ASSERT_EQ(hits[static_cast<std::size_t>(i)], 1) << "i=" << i;
    }
    EXPECT_EQ(ses::parallel_sum(100, 0, [](int i) { return i; }), 4950);

    if (ses::parallel_workers() > 1) {
        std::mutex m;
        std::set<int> seen;
        ses::parallel_ranges(64 * 8, [&](int worker, int, int) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            const std::lock_guard<std::mutex> lk(m);
            seen.insert(worker);
        });
        EXPECT_GT(seen.size(), 1u) << "pool stuck in serial mode after a throw";
    }
}

// Adversarial magnitudes (e^+-30, alt signs): order shows in low bits.
// CONTRACT: the result is the chunk-ordered sum with chunk size ceil(n/64),
// a function of n alone -- so it is bitwise the same at ANY pool width
// (tests/CMakeLists.txt reruns this suite at SES_PARALLEL_WORKERS=1 and 3).
TEST(ParallelSum, BitwiseTheChunkOrderedContractAtAnyWidth) {
    const int n = 4001;
    auto term = [](int i) {
        const double m = std::exp(30.0 * std::sin(0.7 * i));
        return (i % 2 == 0) ? m : -m;
    };
    const int chunk = (n + 63) / 64;
    double contract = 0.0;
    for (int begin = 0; begin < n; begin += chunk) {
        double partial = 0.0;
        for (int i = begin; i < std::min(begin + chunk, n); ++i) {
            partial += term(i);
        }
        contract += partial;
    }
    const double first = ses::parallel_sum(n, 0.0, term);
    EXPECT_EQ(first, contract);  // bitwise, not approx
    for (int rep = 0; rep < 50; ++rep) {
        const double again = ses::parallel_sum(n, 0.0, term);
        ASSERT_EQ(first, again) << "rep=" << rep;
    }
    double serial = 0.0;
    for (int i = 0; i < n; ++i) {
        serial += term(i);
    }
    EXPECT_NEAR(first, serial, 1e-9 * std::abs(serial));
}

TEST(ParallelSum, ComplexAccumulatorAndEmptyRange) {
    const int n = 513;
    auto term = [](int i) {
        return std::complex<double>{std::cos(0.1 * i), std::sin(0.1 * i)};
    };
    const std::complex<double> par = ses::parallel_sum(n, std::complex<double>{}, term);
    std::complex<double> serial{};
    for (int i = 0; i < n; ++i) {
        serial += term(i);
    }
    EXPECT_NEAR(par.real(), serial.real(), 1e-12);
    EXPECT_NEAR(par.imag(), serial.imag(), 1e-12);
    EXPECT_EQ(ses::parallel_sum(0, 42.0, [](int) { return 1.0; }), 42.0);
}

// gtest assertions are not thread-safe on every platform (Windows), so the
// bodies only RECORD; every check runs on the main thread afterwards.
TEST(ParallelRanges, DisjointCoverageAndWorkerIndexBounds) {
    const int n = 12345;
    const int workers = ses::parallel_workers();
    std::vector<int> hits(static_cast<std::size_t>(n), 0);
    std::atomic<int> bad_worker{0};
    std::atomic<int> empty_range{0};
    ses::parallel_ranges(n, [&](int worker, int begin, int end) {
        if (worker < 0 || worker >= workers) {
            ++bad_worker;
            return;
        }
        if (begin >= end) {
            ++empty_range;
        }
        for (int i = begin; i < end; ++i) {
            ++hits[static_cast<std::size_t>(i)];
        }
    });
    EXPECT_EQ(bad_worker.load(), 0);
    EXPECT_EQ(empty_range.load(), 0);
    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(hits[static_cast<std::size_t>(i)], 1) << "i=" << i;
    }
    std::atomic<int> calls{0};
    ses::parallel_ranges(0, [&](int, int, int) { ++calls; });
    EXPECT_EQ(calls.load(), 0) << "n=0 must not call body";
}

}  // namespace
