// RED: Anderson localization contract -- deterministic bounded sub-energy
// landscape, yet transport HALTS (coherent backscattering) vs ballistic W=0 twin.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <vector>

import ses.scenario.anderson1d_director;
import ses.field;
import ses.grid;
import ses.observables;
import ses.propagator;
import ses.wavepacket;

namespace {

TEST(Anderson1D, LandscapeIsDeterministicBoundedAndSubEnergy) {
    const ses::Grid1D g{-60.0, 60.0, 4096};
    const std::vector<double> a = ses_shell::anderson_potential(g, 1.0, 7);
    const std::vector<double> b = ses_shell::anderson_potential(g, 1.0, 7);
    const std::vector<double> c = ses_shell::anderson_potential(g, 1.0, 8);
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    double vmax = 0.0;
    double vsum = 0.0;
    for (const double v : a) {
        vmax = std::max(vmax, std::abs(v));
        vsum += std::abs(v);
    }
    EXPECT_GT(vsum, 0.0);
    // speckle overlap peaks ~1.4x the w=1.0 grain range
    EXPECT_LT(vmax, 1.4);
}

TEST(Anderson1D, DisorderBlocksTheBallisticPacket) {
    // OPEN wire: edge CAPs absorb exits -- a periodic FFT box would wrap the
    // transmitted tail back into the readout. Packet E is above every barrier.
    const ses::Grid1D g{-60.0, 60.0, 4096};
    const double x0 = -45.0;
    const double w0 = 4.0;
    const double cap_w = 6.0;
    std::vector<double> cap(static_cast<std::size_t>(g.n), 1.0);
    for (int i = 0; i < g.n; ++i) {
        const double d = std::min(g.coord(i) - g.xmin,
                                  g.xmax - g.coord(i));
        if (d < cap_w) {
            const double t = 1.0 - d / cap_w;
            cap[static_cast<std::size_t>(i)] =
                std::exp(-w0 * t * t * 0.01);
        }
    }
    // transmitted = flux the RIGHT cap (x>0) absorbs; reflected exits LEFT, uncounted
    auto run = [&](const std::vector<double>& v) {
        const ses::SplitOperator1D prop{g, v, 0.01};
        ses::Field1D psi =
            ses::gaussian_wavepacket(g, x0, 2.0, ses_shell::kAn1dK0);
        const double h = g.spacing();
        double transmitted = 0.0;
        for (int s = 0; s < 11000; ++s) {  // t = 110: full transit + tail
            prop.step(psi, 1);
            for (int i = 0; i < g.n; ++i) {
                const double c = cap[static_cast<std::size_t>(i)];
                if (c < 1.0 && g.coord(i) > 0.0) {
                    transmitted += std::norm(psi[i]) * (1.0 - c * c) * h;
                }
                psi[i] *= c;
            }
        }
        return transmitted;
    };
    const std::vector<double> clean(static_cast<std::size_t>(g.n), 0.0);
    const std::vector<double> dis =
        ses_shell::anderson_potential(g, ses_shell::kAn1dW, 7);
    const double t_clean = run(clean);
    const double t_dis = run(dis);
    std::printf("anderson: transmitted clean %.3f, disordered %.3f\n",
                t_clean, t_dis);
    EXPECT_GT(t_clean, 0.7);
    EXPECT_LT(t_dis, 0.3 * t_clean);
}

}  // namespace
