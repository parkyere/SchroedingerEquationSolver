// DoubleSlit2DDirector contracts (needs the GPU-layer library ses_scenario).

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

import ses.scenario;
import ses.scenario.doubleslit2d_director;

namespace {

struct P3 {
    float x, y, z;
};

// The solenoid overlay ends in its flux arrow: tail, tip, left barb, tip,
// right barb (the core circle comes first).
struct Arrow {
    P3 tail, tip, left, right;
};
Arrow solenoid_arrow(const ses_shell::DoubleSlit2DDirector& d) {
    const ses_shell::OverlayCurve c = d.overlay_curve(1);
    const auto at = [&c](int i) {
        return P3{c.xyz[3 * i], c.xyz[3 * i + 1], c.xyz[3 * i + 2]};
    };
    const int n = c.count;
    return {at(n - 5), at(n - 4), at(n - 3), at(n - 1)};
}

// Signed arrow length is 5 Phi / 2 pi; with no flux there is no direction to
// show, so no arrowhead may be drawn (it used to point up at Phi = 0).
TEST(DoubleSlit2DDirector, SolenoidArrowHasNoHeadWithoutFlux) {
    ses_shell::DoubleSlit2DDirector d;
    ASSERT_EQ(d.slit()->flux(), 0.0);
    const Arrow a = solenoid_arrow(d);
    EXPECT_EQ(a.tip.z, 0.0f);
    for (const P3& barb : {a.left, a.right}) {
        EXPECT_FLOAT_EQ(barb.x, a.tip.x);
        EXPECT_FLOAT_EQ(barb.z, a.tip.z);
    }
}

// The head may not be longer than the shaft: for a small flux its base used
// to sit 0.4 bohr on the far side of the tail.
TEST(DoubleSlit2DDirector, SolenoidArrowHeadStaysOnTheShaft) {
    ses_shell::DoubleSlit2DDirector d;
    for (const double phi : {0.3, -0.3}) {
        d.slit()->set_flux(phi);
        const Arrow a = solenoid_arrow(d);
        const float len = a.tip.z - a.tail.z;
        ASSERT_NE(len, 0.0f);
        for (const P3& barb : {a.left, a.right}) {
            const float along = (barb.z - a.tail.z) / len;  // 0 tail .. 1 tip
            EXPECT_GE(along, 0.0f) << "Phi = " << phi;
            EXPECT_LE(along, 1.0f) << "Phi = " << phi;
        }
    }
}

// A full flux quantum keeps the established look: 5 bohr shaft, head base at
// len - 0.18 len - 0.6, barbs 0.8 bohr out.
TEST(DoubleSlit2DDirector, SolenoidArrowAtOneFluxQuantum) {
    ses_shell::DoubleSlit2DDirector d;
    d.slit()->set_flux(2.0 * std::numbers::pi);
    const Arrow a = solenoid_arrow(d);
    EXPECT_NEAR(a.tip.z, 5.0f, 1e-4f);
    EXPECT_NEAR(a.left.z, 3.5f, 1e-4f);
    EXPECT_NEAR(a.right.z, 3.5f, 1e-4f);
    EXPECT_NEAR(a.tip.x - a.left.x, 0.8f, 1e-4f);
    EXPECT_NEAR(a.right.x - a.tip.x, 0.8f, 1e-4f);
}

}  // namespace
