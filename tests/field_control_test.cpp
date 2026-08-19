// Static uniform E / B field controls behind the hydrogen panel sliders and
// the director's field gates. The user must be able to point either field
// in EITHER direction ALONG ANY AXIS: a negative value is a real, active field
// (E -> -E flips the potential tilt, B -> -B reverses the Larmor sense), so
// "field is on" means != 0, never > 0; and both fields carry a coordinate
// axis (0=x,1=y,2=z), not a hardwired +z. Contract for
// scenario/src/field_control.ixx.

#include <gtest/gtest.h>

import ses.scenario.field_control;

namespace {

using ses_shell::FieldControl;
using ses_shell::field_active;

TEST(FieldControl, BothFieldsCarryAnAxisDefaultingToZ) {
    const FieldControl f{};
    EXPECT_EQ(f.e_axis, 2);
    EXPECT_EQ(f.b_axis, 2);
}

TEST(FieldControl, AxisCyclesXYZAndWraps) {
    FieldControl f{};
    f.cycle_e_axis();  // z -> x
    EXPECT_EQ(f.e_axis, 0);
    f.cycle_e_axis();  // x -> y
    EXPECT_EQ(f.e_axis, 1);
    f.cycle_e_axis();  // y -> z
    EXPECT_EQ(f.e_axis, 2);
    // Independent of B's axis.
    EXPECT_EQ(f.b_axis, 2);
    f.cycle_b_axis();
    EXPECT_EQ(f.b_axis, 0);
    EXPECT_EQ(f.e_axis, 2);
}

TEST(FieldControl, AxisNameIsXYZ) {
    EXPECT_STREQ(FieldControl::axis_name(0), "x");
    EXPECT_STREQ(FieldControl::axis_name(1), "y");
    EXPECT_STREQ(FieldControl::axis_name(2), "z");
}

TEST(FieldControl, ZeroIsOff) {
    EXPECT_FALSE(field_active(0.0));
    EXPECT_FALSE(field_active(-0.0));
}

TEST(FieldControl, PositiveIsOn) {
    EXPECT_TRUE(field_active(0.05));
    EXPECT_TRUE(field_active(1e-9));
}

TEST(FieldControl, NegativeIsOnToo) {
    // The defect this pins: a > 0 gate silently treats -0.05 as "no field".
    EXPECT_TRUE(field_active(-0.05));
    EXPECT_TRUE(field_active(-1e-9));
}

TEST(FieldControl, DefaultsAreBothOff) {
    const FieldControl f{};
    EXPECT_DOUBLE_EQ(f.e0, 0.0);
    EXPECT_DOUBLE_EQ(f.b, 0.0);
    EXPECT_FALSE(f.e_active());
    EXPECT_FALSE(f.b_active());
    EXPECT_FALSE(f.any_active());
}

TEST(FieldControl, EitherSignActivatesEachField) {
    FieldControl f{};
    f.e0 = -0.05;
    EXPECT_TRUE(f.e_active());
    EXPECT_FALSE(f.b_active());
    EXPECT_TRUE(f.any_active());
    f.e0 = 0.0;
    f.b = -0.2;
    EXPECT_FALSE(f.e_active());
    EXPECT_TRUE(f.b_active());
    EXPECT_TRUE(f.any_active());
}

// The E tilt E*z is odd in E: the potential added for -E is the negative of
// the one added for +E, so the field really points the other way.
TEST(FieldControl, ETiltIsOddInE) {
    EXPECT_DOUBLE_EQ(FieldControl::e_tilt(-0.05, 3.0),
                     -FieldControl::e_tilt(0.05, 3.0));
    EXPECT_DOUBLE_EQ(FieldControl::e_tilt(0.05, 3.0), 0.15);
    EXPECT_DOUBLE_EQ(FieldControl::e_tilt(0.05, -3.0), -0.15);
}

}  // namespace
