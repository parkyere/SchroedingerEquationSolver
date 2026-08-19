module;
export module ses.scenario.field_control;

// Static uniform E / B field controls (hydrogen scene): SIGNED magnitude +
// coordinate axis for each. Sign is physical (E -> -E flips the potential
// tilt; B -> -B reverses the Larmor sense) so "on" is != 0, never > 0.
// CONTRACT: tests/field_control_test.cpp.

export namespace ses_shell {

// A field is active in EITHER direction; only exactly zero is "off".
constexpr bool field_active(double x) noexcept { return x != 0.0; }

struct FieldControl {
    double e0 = 0.0;  // E magnitude (au), signed; 0 = off
    double b = 0.0;   // B magnitude (au), signed; 0 = off
    int e_axis = 2;   // 0=x 1=y 2=z
    int b_axis = 2;

    constexpr bool e_active() const noexcept { return field_active(e0); }
    constexpr bool b_active() const noexcept { return field_active(b); }
    constexpr bool any_active() const noexcept { return e_active() || b_active(); }

    // z -> x -> y -> z (matches the existing B-axis toggle order).
    void cycle_e_axis() noexcept { e_axis = (e_axis + 1) % 3; }
    void cycle_b_axis() noexcept { b_axis = (b_axis + 1) % 3; }

    // Potential tilt of a uniform E along the chosen axis: E * (coordinate
    // along that axis). Odd in E -- the field really points the other way.
    static constexpr double e_tilt(double e0, double coord_along_axis) noexcept {
        return e0 * coord_along_axis;
    }

    static constexpr const char* axis_name(int axis) noexcept {
        return axis == 0 ? "x" : (axis == 1 ? "y" : "z");
    }
};

}  // namespace ses_shell
