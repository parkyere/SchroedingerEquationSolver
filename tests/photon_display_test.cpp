// Photon display policy: streak helix (twist sense = helicity, comoving
// phase), constant display speed (Lyman-alpha = 2 s), flash 1/3 peak, and the
// 16-slot concurrent-flight pool. Contract for core/src/photon_display.ixx.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
import ses.vec;
import ses.photon_display;

namespace {

using ses::Vec3d;

// Flash: linear decay with the peak capped at 1/3 (full bright hurt the eye).

TEST(FlashIntensity, PeakIsOneThirdAndDecaysLinearly) {
    EXPECT_NEAR(ses::flash_intensity(25, 25), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(ses::flash_intensity(12, 24), 1.0 / 6.0, 1e-12);
    EXPECT_DOUBLE_EQ(ses::flash_intensity(0, 25), 0.0);
    EXPECT_DOUBLE_EQ(ses::flash_intensity(5, 0), 0.0);  // guard
}

// Flight pool: 16 concurrent streaks (3 overlapping 5-photon cascades);
// overflow evicts the most-spent.

TEST(PhotonFlightPool, CapacityCoversThreeOverlappingCascades) {
    EXPECT_EQ(ses::kMaxPhotonFlights, 16);  // 3 x 5-photon Yrast + margin
    ses::PhotonFlightPool pool;
    for (int i = 0; i < ses::kMaxPhotonFlights; ++i) {
        pool.spawn(ses::PhotonRecord{Vec3d{0.0, 0.0, 1.0}, +1}, 0.375, 120);
        EXPECT_EQ(pool.count(), i + 1);
    }
}

TEST(PhotonFlightPool, OverflowEvictsTheMostSpentFlight) {
    ses::PhotonFlightPool pool;
    // Tag flights by delta_e; ages diverge via interleaved advances.
    pool.spawn(ses::PhotonRecord{Vec3d{0.0, 0.0, 1.0}, +1}, 0.1, 100);
    for (int t = 0; t < 50; ++t) {
        pool.advance();  // flight 0.1 at progress 0.5
    }
    pool.spawn(ses::PhotonRecord{Vec3d{0.0, 0.0, 1.0}, +1}, 0.2, 100);
    for (int t = 0; t < 10; ++t) {
        pool.advance();  // 0.1 -> 0.6, 0.2 -> 0.1
    }
    for (int i = 2; i < ses::kMaxPhotonFlights; ++i) {
        // Filler tags out of the 0.x band (0.01*10 rounds to exactly 0.1).
        pool.spawn(ses::PhotonRecord{Vec3d{0.0, 0.0, 1.0}, +1},
                   10.0 + static_cast<double>(i), 100);
    }
    ASSERT_EQ(pool.count(), ses::kMaxPhotonFlights);
    // Overflow spawn: 0.1 is the most spent and must be the one evicted.
    pool.spawn(ses::PhotonRecord{Vec3d{0.0, 0.0, 1.0}, +1}, 0.5, 100);
    ASSERT_EQ(pool.count(), ses::kMaxPhotonFlights);
    bool saw_01 = false;
    bool saw_05 = false;
    for (int i = 0; i < pool.count(); ++i) {
        const ses::PhotonFlightPool::Flight* f = pool.active(i);
        ASSERT_NE(f, nullptr);
        saw_01 = saw_01 || f->delta_e == 0.1;
        saw_05 = saw_05 || f->delta_e == 0.5;
    }
    EXPECT_FALSE(saw_01);
    EXPECT_TRUE(saw_05);
}

TEST(PhotonFlightPool, AdvanceExpiresFlightsPastTheirTotal) {
    ses::PhotonFlightPool pool;
    pool.spawn(ses::PhotonRecord{Vec3d{0.0, 0.0, 1.0}, +1}, 0.375, 2);
    pool.spawn(ses::PhotonRecord{Vec3d{0.0, 0.0, 1.0}, -1}, 0.375, 120);
    pool.advance();
    pool.advance();
    EXPECT_EQ(pool.count(), 2);  // age == total: still drawn
    pool.advance();
    EXPECT_EQ(pool.count(), 1);  // short flight expired past its total
    const ses::PhotonFlightPool::Flight* f = pool.active(0);
    ASSERT_NE(f, nullptr);
    EXPECT_EQ(f->rec.helicity, -1);  // the long flight survives
}

// Photon streak: helix e^{i lambda k s} in the (theta_hat, phi_hat) frame,
// comoving phase (crests ride with the head), ~2 s wall flight, late fade.

TEST(PhotonStreak, FlightConstantsMatchTheUserSpec) {
    EXPECT_EQ(ses::kPhotonFlightTicks, 120);  // Lyman-alpha baseline: 2 s
    // Must exit the +-80 box even along the corner diagonal.
    EXPECT_GE(ses::kPhotonTravel, std::sqrt(3.0) * 80.0);
}

TEST(PhotonStreak, TravelAddsTheFourWavelengthTail) {
    for (const double de : {0.375, 0.75, 0.0694}) {
        EXPECT_DOUBLE_EQ(ses::photon_travel(de),
                         ses::kPhotonTravel +
                             ses::kPhotonTurns *
                                 ses::photon_display_wavelength(de));
    }
}

TEST(PhotonStreak, LymanAlphaTwoSecondsOthersSameDisplaySpeed) {
    EXPECT_EQ(ses::photon_flight_frames(0.375), ses::kPhotonFlightTicks);
    // Constant speed: frames scale with travel (round to whole frames).
    for (const double de : {0.75, 0.486, 0.0694}) {
        const double want = ses::kPhotonFlightTicks * ses::photon_travel(de) /
                            ses::photon_travel(0.375);
        EXPECT_NEAR(ses::photon_flight_frames(de), want, 0.51) << de;
    }
    // Softer than Lyman-alpha flies farther -> lives longer; harder shorter.
    EXPECT_GT(ses::photon_flight_frames(0.0694), ses::kPhotonFlightTicks);
    EXPECT_LT(ses::photon_flight_frames(0.75), ses::kPhotonFlightTicks);
}

TEST(PhotonStreak, WavelengthIsInverseInEnergy) {
    EXPECT_DOUBLE_EQ(ses::photon_display_wavelength(0.375), 25.0);
    EXPECT_DOUBLE_EQ(ses::photon_display_wavelength(0.75), 12.5);
}

TEST(PhotonStreak, AlphaHoldsThenFadesToZero) {
    EXPECT_DOUBLE_EQ(ses::photon_streak_alpha(0.0), 1.0);
    EXPECT_DOUBLE_EQ(ses::photon_streak_alpha(0.5), 1.0);
    EXPECT_LT(ses::photon_streak_alpha(0.95), ses::photon_streak_alpha(0.75));
    EXPECT_DOUBLE_EQ(ses::photon_streak_alpha(1.0), 0.0);
    double prev = 1.0;
    for (int i = 0; i <= 20; ++i) {
        const double a = ses::photon_streak_alpha(i / 20.0);
        EXPECT_LE(a, prev + 1e-12);
        prev = a;
    }
}

TEST(PhotonStreak, HelixTwistSenseFollowsHelicity) {
    const double de = 0.375;  // lambda_display = 25 Bohr, tail = 100
    for (const int lam : {+1, -1}) {
        const ses::PhotonRecord ph{Vec3d{0.0, 0.0, 1.0}, lam};
        const auto v = ses::photon_streak_vertices(ph, de, 0.5);
        ASSERT_EQ(v.size(),
                  static_cast<std::size_t>(ses::kPhotonStreakPoints));
        const double sh = 0.5 * ses::photon_travel(de);
        const std::size_t body = v.size() - 1;
        double prev_s = -1.0;
        for (std::size_t i = 0; i < body; ++i) {
            const double s = v[i].z;  // n = z: axial coordinate
            EXPECT_GT(s, prev_s);
            prev_s = s;
            const double tr =
                std::sqrt(v[i].x * v[i].x + v[i].y * v[i].y);
            EXPECT_NEAR(tr, ses::kPhotonStreakRadius, 1e-9);
        }
        // Comoving phase: at the head s = sh the transverse is +R x_hat...
        const Vec3d& head = v[body - 1];
        EXPECT_NEAR(head.z, sh, 1e-9);
        EXPECT_NEAR(head.x, ses::kPhotonStreakRadius, 1e-9);
        EXPECT_NEAR(head.y, 0.0, 1e-9);
        // ...and the fully-unspooled tail start sits 4 whole turns behind:
        // same phase, same +R x_hat transverse.
        EXPECT_NEAR(v[0].z, sh - ses::kPhotonTurns * 25.0, 1e-9);
        EXPECT_NEAR(v[0].x, ses::kPhotonStreakRadius, 1e-9);
        EXPECT_NEAR(v[0].y, 0.0, 1e-9);
        // Twist sense: consecutive transverse vectors rotate with sign lam.
        for (std::size_t i = 0; i + 1 < body; ++i) {
            const double cross = v[i].x * v[i + 1].y - v[i].y * v[i + 1].x;
            EXPECT_GT(lam * cross, 0.0);
        }
    }
}

TEST(PhotonStreak, TipIsOnAxisAheadAndTailAnchorsAtTheNucleus) {
    const ses::PhotonRecord ph{Vec3d{0.0, 0.0, 1.0}, +1};
    // Early flight (head < tail length): still anchored at the nucleus.
    const auto early = ses::photon_streak_vertices(ph, 0.375, 0.1);
    ASSERT_FALSE(early.empty());
    EXPECT_NEAR(early.front().z, 0.0, 1e-9);
    // The tip is on-axis, strictly ahead of every body vertex.
    const Vec3d tip = early.back();
    EXPECT_NEAR(std::sqrt(tip.x * tip.x + tip.y * tip.y), 0.0, 1e-9);
    for (std::size_t i = 0; i + 1 < early.size(); ++i) {
        EXPECT_LT(early[i].z, tip.z);
    }
    // Full flight reaches past the wavelength-scaled travel.
    const auto done = ses::photon_streak_vertices(ph, 0.375, 1.0);
    ASSERT_FALSE(done.empty());
    EXPECT_GE(done.back().z, ses::photon_travel(0.375));
}

}  // namespace
