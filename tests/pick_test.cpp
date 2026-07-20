// RED: z = 0 stage-plane pick must round-trip through perspective * look_at.

#include <gtest/gtest.h>

#include <cmath>

import ses.camera;
import ses.vec;

namespace {

// column-major m[c*4 + r] -- the pinned Mat4 convention.
void mul4(const ses::Mat4& m, const double v[4], double out[4]) {
    for (int r = 0; r < 4; ++r) {
        out[r] = 0.0;
        for (int c = 0; c < 4; ++c) {
            out[r] += m.m[c * 4 + r] * v[c];
        }
    }
}

TEST(PickPlane, CenterHitsTheOriginAndPointsRoundTrip) {
    const double kPi = 3.14159265358979323846;
    const double az = 0.35;
    const double el = 0.95;
    const double dist = 75.0;
    const double fovy = 45.0 * kPi / 180.0;
    const double aspect = 16.0 / 9.0;

    double x = 1e9;
    double y = 1e9;
    ASSERT_TRUE(ses::unproject_to_z0(az, el, dist, fovy, aspect, 0.0, 0.0,
                                     &x, &y));
    EXPECT_NEAR(x, 0.0, 1e-9);  // orbit camera looks at the origin
    EXPECT_NEAR(y, 0.0, 1e-9);

    const ses::Mat4 proj = ses::perspective(fovy, aspect, 0.1, 500.0);
    const ses::Vec3d eye = ses::orbit_eye(az, el, dist, ses::Vec3d{});
    const ses::Mat4 view =
        ses::look_at(eye, ses::Vec3d{}, ses::Vec3d{0.0, 1.0, 0.0});
    const ses::Mat4 mvp = proj * view;
    const double wpt[4] = {7.0, -4.0, 0.0, 1.0};
    double clip[4];
    mul4(mvp, wpt, clip);
    ASSERT_GT(clip[3], 0.0);
    const double ndc_x = clip[0] / clip[3];
    const double ndc_y = clip[1] / clip[3];
    ASSERT_TRUE(ses::unproject_to_z0(az, el, dist, fovy, aspect, ndc_x,
                                     ndc_y, &x, &y));
    EXPECT_NEAR(x, 7.0, 1e-6);
    EXPECT_NEAR(y, -4.0, 1e-6);

    // Degenerate: elevation pi/2 => ray lies in z = 0; must refuse, not divide by zero.
    EXPECT_FALSE(ses::unproject_to_z0(0.0, 0.5 * kPi, dist, fovy, aspect,
                                      0.0, 0.0, &x, &y));
}

}  // namespace
