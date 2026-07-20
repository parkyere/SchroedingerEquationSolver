// RED: IBM-corral pieces the scene simplified away; the corral director copies
// these (its CONTRACT comments point back here). Provenance: m* = 0.38 Cu(111)
// surface state; black-dot fence ~50% absorbers (Crommie).

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
import ses.grid;
import ses.vec;
import ses.field;
import ses.potential;
import ses.propagator;
import ses.imaginary_time;
import ses.observables;
import ses.wavepacket;
import ses.parallel;
import ses.scenario.corral2d_director;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Strong non-leaky ring: the metastable disc mode tunnels out faster at
// m* = 0.38, so both masses plateau; the leaky-fence relax is the arc's job.
std::vector<double> ring_wall(const ses::Grid3D& g, double radius, double amp,
                              double width) {
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            const double r = std::sqrt(x * x + y * y);
            if (r >= radius && r < radius + width) {
                v[static_cast<std::size_t>(g.flat(i, j, 0))] = amp;
            }
        }
    }
    return v;
}

// Metastable disc mode drains to the outside-box ground, so a convergence
// gate loses for small m*: use a FIXED tau = 20 on the plateau (settled by
// ~16, crossover past ~45).
double relax_ground_energy(const ses::Grid3D& g, const std::vector<double>& v,
                           double mass) {
    ses::Field3D psi = ses::gaussian_wavepacket(
        g, ses::Vec3d{0.5, 0.0, 0.0}, ses::Vec3d{2.0, 2.0, 0.5},
        ses::Vec3d{0.0, 0.0, 0.0});
    const ses::ImaginaryTimePropagator3D itp{g, v, 0.02, mass};
    itp.relax(psi, 1000);  // tau = 20
    return ses::mean_energy(psi, v, mass);
}

TEST(Corral2D, EffectiveMassScalesTheGroundToJZeroBand) {
    const double r = 6.0;
    const ses::Grid3D g{ses::Grid1D{-16.0, 16.0, 128},
                        ses::Grid1D{-16.0, 16.0, 128}, ses::Grid1D{0.0, 2.0, 1}};
    const std::vector<double> v = ring_wall(g, r, 4.0, 3.0);
    const double mstar = 0.38;
    const double e_star = relax_ground_energy(g, v, mstar);
    const double e_unit = relax_ground_energy(g, v, 1.0);
    // Hard-disk J0 ground band, same as the arc.
    const double j01 = 2.405;
    const double hard_star = j01 * j01 / (2.0 * mstar * r * r);
    EXPECT_GT(e_star, 0.6 * hard_star);
    EXPECT_LT(e_star, 2.0 * hard_star);
    // E0 ~ 1/m (near-free interior).
    EXPECT_NEAR(e_star * mstar / e_unit, 1.0, 0.15);
}

TEST(Corral2D, OpenBoundaryKillsThePeriodicRevival) {
    const ses::Grid3D g{ses::Grid1D{-16.0, 16.0, 128},
                        ses::Grid1D{-16.0, 16.0, 128}, ses::Grid1D{0.0, 2.0, 1}};
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const std::vector<double> mask = ses::absorbing_mask(g, 3.0);
    const double k0 = 3.0;
    const double dt = 0.05;
    const int nsteps = 240;  // t = 12 au covers the wrap-return (~10.7)
    auto p_center = [&](const ses::Field3D& f) {
        double inside = 0.0;
        double total = 0.0;
        for (int j = 0; j < g.y.n; ++j) {
            const double y = g.y.coord(j);
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double w = std::norm(f(i, j, 0));
                total += w;
                if (x * x + y * y < 5.0 * 5.0) {
                    inside += w;
                }
            }
        }
        return total > 0.0 ? inside / total : 0.0;
    };
    const auto packet = [&] {
        return ses::gaussian_wavepacket(g, ses::Vec3d{0.0, 0.0, 0.0},
                                        ses::Vec3d{1.5, 1.5, 0.5},
                                        ses::Vec3d{k0, 0.0, 0.0});
    };
    const ses::SplitOperator3D prop{g, v, dt};

    // Periodic (no mask): wraps around and revives at center.
    ses::Field3D per = packet();
    prop.step(per, nsteps);
    EXPECT_GT(p_center(per), 0.25);

    // Open boundary (per-step mask): leaked flux gone, then conditional renorm.
    ses::Field3D open = packet();
    for (int s = 0; s < nsteps; ++s) {
        prop.step(open, 1);
        for (std::size_t i = 0; i < open.data().size(); ++i) {
            open.data()[i] *= mask[i];
        }
    }
    EXPECT_LT(ses::norm_sq(open), 0.05);
    ses::normalize(open);
    if (ses::norm_sq(open) > 0.0) {
        EXPECT_NEAR(ses::norm_sq(open), 1.0, 1e-9);
    }
}

TEST(Corral2D, BlackDotFenceAbsorbsPartially) {
    const ses::Grid3D g{ses::Grid1D{-32.0, 32.0, 256}, ses::Grid1D{0.0, 2.0, 1},
                        ses::Grid1D{0.0, 2.0, 1}};
    const double amp = 1.5;
    const double sigma = 0.6;
    const double w0 = 0.8;  // black-dot strength (peak of W)
    const double dt = 0.05;
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    std::vector<double> damp(static_cast<std::size_t>(g.size()), 1.0);
    for (int i = 0; i < g.x.n; ++i) {
        const double x = g.x.coord(i);
        const double bump = amp * std::exp(-x * x / (2.0 * sigma * sigma));
        v[static_cast<std::size_t>(g.flat(i, 0, 0))] = bump;
        damp[static_cast<std::size_t>(g.flat(i, 0, 0))] =
            std::exp(-w0 * (bump / amp) * dt);
    }
    ses::Field3D psi = ses::gaussian_wavepacket(
        g, ses::Vec3d{-10.0, 0.0, 0.0}, ses::Vec3d{2.0, 0.5, 0.5},
        ses::Vec3d{2.5, 0.0, 0.0});  // E_k ~ 3.1 > barrier 1.5: mostly crosses
    const ses::SplitOperator3D prop{g, v, dt};
    for (int s = 0; s < 200; ++s) {  // t = 10: fully past the bump
        prop.step(psi, 1);
        for (std::size_t i = 0; i < psi.data().size(); ++i) {
            psi.data()[i] *= damp[i];
        }
    }
    const double survived = ses::norm_sq(psi);
    EXPECT_LT(survived, 0.90);
    EXPECT_GT(survived, 0.05);
}

// fermi_wave() = standing wave at E_F (k_F R = j0_10, ~10 nodes): the 1993
// topograph images E_F LDOS, NOT the ground.
TEST(Corral2DDirector, FermiWaveIsQuasiStationary) {
    ses_shell::Corral2DDirector d;
    ses_shell::CorralApi* api = d.corral();
    ASSERT_NE(api, nullptr);
    api->fermi_wave();
    const double conf0 = api->confinement();
    EXPECT_GT(conf0, 0.75);  // J0 tail reaches past R by construction
    for (int t = 0; t < 30; ++t) {
        d.tick();
        d.run_frame();
    }
    // measured ~86% over ~5 au; the fence absorbs by design, floor 0.8.
    EXPECT_GT(api->confinement(), 0.8 * conf0);
}

}  // namespace
