#pragma once

// Minimal 3-component double vector. Deliberately a bare aggregate: operators
// and quaternion math arrive test-first when the camera work (Phase 7)
// demands them.

namespace ses {

struct Vec3d {
    double x{};
    double y{};
    double z{};
};

}  // namespace ses
