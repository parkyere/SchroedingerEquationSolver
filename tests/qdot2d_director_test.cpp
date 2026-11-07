// Qdot2DDirector contracts (needs the GPU-layer library ses_scenario).

#include <gtest/gtest.h>

#include <cmath>

import ses.scenario.qdot2d_director;

namespace {

// The lattice is in Landau gauge (A_x = +B y). A packet gathered AT REST at
// r0 is the displaced Fock-Darwin ground state, energy Omega + w0^2 r0^2/2;
// a real Gaussian there carries the mechanical momentum pi_x = -B y0 (plus a
// breathing excitation B^2/(8 Omega)), i.e. +4.6 Ha at y0 = 5, B = 0.6.
TEST(Qdot2DDirector, GrabGathersAPacketAtRest) {
    ses_shell::Qdot2DDirector d;
    ses_shell::QdotApi* q = d.qdot();
    ASSERT_NE(q, nullptr);
    const double y0 = 5.0;
    q->begin_grab(0.0, y0);
    q->update_grab(1.0);
    q->end_grab();
    const double w0 = q->omega0();
    const double expected = q->energy_pred() + 0.5 * w0 * w0 * y0 * y0;
    EXPECT_NEAR(q->energy_meas(), expected, 0.05)
        << "B = " << q->field() << ", Omega = " << q->energy_pred();
}

}  // namespace
