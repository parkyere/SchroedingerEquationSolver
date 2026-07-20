// Handedness: e^{+i phi} (m=+1) circulates CCW about +z, matching L_z = +m (ses.measurement).

#include <complex>

#include <gtest/gtest.h>
import ses.vec;
import ses.flow;


namespace {

using ses::Vec3d;

// Plane wave e^{ikx}: j = k|psi|^2, so j_x = k at psi = 1.
TEST(Flow, PlaneWaveCurrentIsKTimesDensity) {
    const double k = 2.0;
    const std::complex<double> psi{1.0, 0.0};
    const std::complex<double> dx{0.0, k};
    const Vec3d j = ses::probability_current(psi, dx, {0.0, 0.0}, {0.0, 0.0});
    EXPECT_NEAR(j.x, k, 1e-12);
    EXPECT_NEAR(j.y, 0.0, 1e-12);
    EXPECT_NEAR(j.z, 0.0, 1e-12);
}

// m = +1 ring psi ~ x + i y: grad = (1, i, 0); at (1,0) current -> +y (CCW, L_z > 0).
TEST(Flow, RingStateM1CirculatesCounterclockwise) {
    const std::complex<double> psi{1.0, 0.0};
    const std::complex<double> dx{1.0, 0.0};
    const std::complex<double> dy{0.0, 1.0};
    const Vec3d j = ses::probability_current(psi, dx, dy, {0.0, 0.0});
    EXPECT_NEAR(j.x, 0.0, 1e-12);
    EXPECT_GT(j.y, 0.0);
}

// m = -1 ring (x - i y): grad = (1, -i, 0); at (1,0) current -> -y (CW, L_z < 0).
TEST(Flow, RingStateMinus1CirculatesClockwise) {
    const std::complex<double> psi{1.0, 0.0};
    const std::complex<double> dx{1.0, 0.0};
    const std::complex<double> dy{0.0, -1.0};
    const Vec3d j = ses::probability_current(psi, dx, dy, {0.0, 0.0});
    EXPECT_LT(j.y, 0.0);
}

TEST(Flow, RealStateHasNoCurrent) {
    const std::complex<double> psi{0.7, 0.0};
    const std::complex<double> dx{-0.3, 0.0};
    const std::complex<double> dy{0.5, 0.0};
    const std::complex<double> dz{0.1, 0.0};
    const Vec3d j = ses::probability_current(psi, dx, dy, dz);
    EXPECT_NEAR(j.x, 0.0, 1e-15);
    EXPECT_NEAR(j.y, 0.0, 1e-15);
    EXPECT_NEAR(j.z, 0.0, 1e-15);
}

// v = j/rho; guarded to 0 at a node (rho -> 0).
TEST(Flow, BohmianVelocityGuardsNodes) {
    const std::complex<double> zero{0.0, 0.0};
    const std::complex<double> dx{5.0, 5.0};
    const Vec3d v = ses::bohmian_velocity(zero, dx, {0.0, 0.0}, {0.0, 0.0});
    EXPECT_EQ(v.x, 0.0);
    EXPECT_EQ(v.y, 0.0);
    EXPECT_EQ(v.z, 0.0);
}

TEST(Flow, TrailFadeMonotoneTailToHead) {
    EXPECT_NEAR(ses::trail_fade(0, 40), 0.0, 1e-12);
    EXPECT_NEAR(ses::trail_fade(39, 40), 1.0, 1e-12);
    EXPECT_LT(ses::trail_fade(10, 40), ses::trail_fade(20, 40));
    EXPECT_DOUBLE_EQ(ses::trail_fade(0, 1), 1.0);  // degenerate length
}

}  // namespace
