// The 2D Peierls lattice propagator: engine for the double-slit + Aharonov-Bohm
// scene. WHY a lattice not FFT: split-operator cannot Trotterize (p - A)^2/2 for
// a LOCALIZED flux; on the lattice the flux rides the hopping links as exact
// Peierls phases (unitary to round-off). Gauge: string cut runs +y from the
// solenoid, only crossed x-links carry e^{+-i Phi}.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <numbers>
#include <utility>
#include <vector>

import ses.field;
import ses.grid;
import ses.lattice2d;
import ses.potential;
import ses.scenario.landau2d_director;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;

Grid3D plane_grid(double lx, int nx, double ly, int ny) {
    return Grid3D{Grid1D{-lx, lx, nx}, Grid1D{-ly, ly, ny},
                  Grid1D{-1.0, 1.0, 1}};
}

Field3D plane_packet(const Grid3D& g, double x0, double y0, double sigma,
                     double k0) {
    Field3D psi{g};
    for (int j = 0; j < g.y.n; ++j) {
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            const double y = g.y.coord(j);
            const double env = std::exp(
                -((x - x0) * (x - x0) + (y - y0) * (y - y0)) /
                (4.0 * sigma * sigma));
            psi(i, j, 0) = env * std::complex<double>{std::cos(k0 * x),
                                                      std::sin(k0 * x)};
        }
    }
    ses::normalize(psi);
    return psi;
}

double mean_x(const Field3D& psi) {
    const Grid3D& g = psi.grid();
    double num = 0.0;
    double den = 0.0;
    for (int j = 0; j < g.y.n; ++j) {
        for (int i = 0; i < g.x.n; ++i) {
            const double w = std::norm(psi(i, j, 0));
            num += g.x.coord(i) * w;
            den += w;
        }
    }
    return num / den;
}

TEST(PeierlsLattice2D, IsUnitaryToRoundOff) {
    const Grid3D g = plane_grid(10.0, 64, 10.0, 64);
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    ses::PeierlsLattice2D prop{g, v, 0.02};
    Field3D psi = plane_packet(g, -3.0, 1.0, 2.0, 1.5);
    prop.step(psi, 500);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);
}

TEST(PeierlsLattice2D, FreePacketMovesAtTheLatticeGroupVelocity) {
    const Grid3D g = plane_grid(30.0, 256, 10.0, 32);
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const double dt = 0.01;
    ses::PeierlsLattice2D prop{g, v, dt};
    const double k0 = 1.2;
    Field3D psi = plane_packet(g, -15.0, 0.0, 3.0, k0);
    const double x_start = mean_x(psi);
    const int steps = 1500;
    prop.step(psi, steps);
    const double h = g.x.spacing();
    const double vg = std::sin(k0 * h) / h;
    EXPECT_NEAR(mean_x(psi) - x_start, vg * dt * steps,
                0.01 * vg * dt * steps);
}

TEST(PeierlsLattice2D, SolenoidLinksCarryPureTopology) {
    const Grid3D g = plane_grid(8.0, 32, 8.0, 32);
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    ses::PeierlsLattice2D prop{g, v, 0.02};
    const double phi = 1.9;
    const double xs = g.x.coord(15) + 0.5 * g.x.spacing();
    const double ys = g.y.coord(12) + 0.5 * g.y.spacing();
    prop.set_solenoid(phi, xs, ys);
    int hot = 0;
    for (int j = 0; j + 1 < g.y.n; ++j) {
        for (int i = 0; i + 1 < g.x.n; ++i) {
            // Counter-clockwise around plaquette (i, j):
            const std::complex<double> loop =
                prop.link_x(i, j) * prop.link_y(i + 1, j) *
                std::conj(prop.link_x(i, j + 1)) *
                std::conj(prop.link_y(i, j));
            const double flux = std::arg(loop);
            const bool at_solenoid = i == 15 && j == 12;
            if (at_solenoid) {
                EXPECT_NEAR(std::remainder(flux - phi,
                                           2.0 * std::numbers::pi),
                            0.0, 1e-12);
                ++hot;
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-12)
                    << "stray field at plaquette " << i << "," << j;
            }
        }
    }
    EXPECT_EQ(hot, 1);
}

TEST(PeierlsLattice2D, GaugeCutDirectionIsInvisible) {
    // Down vs up string cut = gauge transform G = diag(e^{-i phi}) on every site
    // RIGHT of the solenoid column; evolve G psi0 (down) and psi0 (up) and every
    // density must match to round-off.
    const Grid3D g = plane_grid(10.0, 64, 10.0, 64);
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const double phi = 2.4;
    const double xs = 0.31;
    const double ys = -0.17;
    ses::PeierlsLattice2D up{g, v, 0.02};
    up.set_solenoid(phi, xs, ys, true);
    ses::PeierlsLattice2D down{g, v, 0.02};
    down.set_solenoid(phi, xs, ys, false);
    Field3D a = plane_packet(g, -3.0, 0.5, 2.0, 1.0);
    Field3D b = a;
    int is = -1;
    for (int i = 0; i + 1 < g.x.n; ++i) {
        if (g.x.coord(i) <= xs && xs < g.x.coord(i + 1)) {
            is = i;
        }
    }
    ASSERT_GE(is, 0);
    const std::complex<double> gph{std::cos(phi), -std::sin(phi)};
    for (int j = 0; j < g.y.n; ++j) {
        for (int i = is + 1; i < g.x.n; ++i) {
            b(i, j, 0) *= gph;
        }
    }
    up.step(a, 400);
    down.step(b, 400);
    for (int j = 0; j < g.y.n; j += 3) {
        for (int i = 0; i < g.x.n; i += 3) {
            EXPECT_NEAR(std::norm(a(i, j, 0)), std::norm(b(i, j, 0)), 1e-10)
                << "at " << i << "," << j;
        }
    }
}

TEST(PeierlsLattice2D, UniformFieldFillsEveryPlaquetteEqually) {
    const Grid3D g = plane_grid(8.0, 32, 8.0, 32);
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    ses::PeierlsLattice2D prop{g, v, 0.02};
    const double b = 0.7;
    prop.set_uniform_field(b);
    const double want = b * g.x.spacing() * g.y.spacing();
    for (int j = 0; j + 1 < g.y.n; ++j) {
        for (int i = 0; i + 1 < g.x.n; ++i) {
            const std::complex<double> loop =
                prop.link_x(i, j) * prop.link_y(i + 1, j) *
                std::conj(prop.link_x(i, j + 1)) *
                std::conj(prop.link_y(i, j));
            EXPECT_NEAR(std::remainder(std::arg(loop) - want,
                                       2.0 * std::numbers::pi),
                        0.0, 1e-12)
                << "plaquette " << i << "," << j;
        }
    }
}

TEST(PeierlsLattice2D, CyclotronOrbitAndLandauRevival) {
    // B = 0.5, k0 = 1: cyclotron radius k0/B = 2, period T = 2 pi / B ~ 12.57;
    // launch at (2, 0) with v = (0, k0) circles the origin -- antipode (-2, 0)
    // at T/2, back at T. The Landau ladder also re-coheres the whole state.
    const Grid3D g = plane_grid(12.0, 96, 12.0, 96);
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const double b = 0.5;
    const double k0 = 1.0;
    const double dt = 0.01;
    ses::PeierlsLattice2D prop{g, v, dt};
    prop.set_uniform_field(b);
    // On the launch row y = 0, A = 0, so the plain plane-wave factor IS the
    // mechanical momentum +k0 in y.
    Field3D psi{g};
    for (int j = 0; j < g.y.n; ++j) {
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            const double y = g.y.coord(j);
            const double env = std::exp(
                -((x - 2.0) * (x - 2.0) + y * y) / (4.0 * 1.4 * 1.4));
            psi(i, j, 0) = env * std::complex<double>{std::cos(k0 * y),
                                                      std::sin(k0 * y)};
        }
    }
    ses::normalize(psi);
    const Field3D start = psi;

    auto mean_r = [&](const Field3D& f) {
        double mx = 0.0;
        double my = 0.0;
        double den = 0.0;
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double w = std::norm(f(i, j, 0));
                mx += g.x.coord(i) * w;
                my += g.y.coord(j) * w;
                den += w;
            }
        }
        return std::pair<double, double>{mx / den, my / den};
    };

    const double period = 2.0 * std::numbers::pi / b;
    const int half = static_cast<int>(0.5 * period / dt + 0.5);
    prop.step(psi, half);
    const auto [hx, hy] = mean_r(psi);
    EXPECT_NEAR(hx, -2.0, 0.4);
    EXPECT_NEAR(hy, 0.0, 0.4);
    prop.step(psi, half);
    const auto [fx, fy] = mean_r(psi);
    EXPECT_NEAR(fx, 2.0, 0.4);
    EXPECT_NEAR(fy, 0.0, 0.4);
    std::complex<double> ov{};
    for (int j = 0; j < g.y.n; ++j) {
        for (int i = 0; i < g.x.n; ++i) {
            ov += std::conj(start(i, j, 0)) * psi(i, j, 0);
        }
    }
    ov *= g.x.spacing() * g.y.spacing() * g.z.spacing();
    EXPECT_GT(std::norm(ov), 0.8);
}

TEST(PeierlsLattice2D, DoubleSlitCarriesTheSolenoidFlux) {
    // Solenoid buried mid-wall: Phi = 0 gives a bright central fringe, Phi = pi
    // kills it, Phi = 2 pi restores it (one flux quantum) -- the AB shift.
    const Grid3D g = plane_grid(24.0, 192, 16.0, 128);
    const double h = g.x.spacing();
    const double wall_lo = 0.0;
    const double wall_hi = 1.5;
    const double sep = 4.0;
    const double width = 1.2;
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        const bool open = std::abs(y - 0.5 * sep) <= 0.5 * width ||
                          std::abs(y + 0.5 * sep) <= 0.5 * width;
        if (open) {
            continue;
        }
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            if (x >= wall_lo && x <= wall_hi) {
                v[static_cast<std::size_t>(g.flat(i, j, 0))] = 40.0;
            }
        }
    }

    auto axis_density = [&](double phi) {
        ses::PeierlsLattice2D prop{g, v, 0.01};
        prop.set_solenoid(phi, 0.5 * (wall_lo + wall_hi) + 0.25 * h, 0.0);
        Field3D psi = plane_packet(g, -12.0, 0.0, 4.0, 2.0);
        prop.step(psi, 1200);
        double sum = 0.0;
        int cnt = 0;
        for (int j = 0; j < g.y.n; ++j) {
            const double y = g.y.coord(j);
            if (std::abs(y) > 0.6) {
                continue;
            }
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                if (x > 6.0 && x < 12.0) {
                    sum += std::norm(psi(i, j, 0));
                    ++cnt;
                }
            }
        }
        return sum / cnt;
    };

    const double bright = axis_density(0.0);
    const double dark = axis_density(std::numbers::pi);
    const double again = axis_density(2.0 * std::numbers::pi);
    EXPECT_GT(bright, 0.0);
    EXPECT_LT(dark, 0.35 * bright);
    EXPECT_NEAR(again, bright, 1e-6 * bright);
}

// Imaginary-time relaxation on the lattice, whose LINK PHASES ride along, so it
// finds a dot ground state IN a magnetic field: Fock-Darwin
// E = Omega = sqrt(w0^2 + B^2/4), unreachable by B = 0 machinery.
TEST(PeierlsLattice2D, RelaxFindsTheFockDarwinGround) {
    const Grid3D g = plane_grid(20.0, 128, 20.0, 128);
    const double w0 = 0.5;
    const double b = 0.6;
    std::vector<double> v(static_cast<std::size_t>(g.size()));
    for (int j = 0; j < g.y.n; ++j) {
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            const double y = g.y.coord(j);
            v[static_cast<std::size_t>(g.flat(i, j, 0))] =
                0.5 * w0 * w0 * (x * x + y * y);
        }
    }
    ses::PeierlsLattice2D prop{g, v, 0.02};
    prop.set_uniform_field(b);
    Field3D psi = plane_packet(g, 1.0, -2.0, 3.0, 0.0);
    prop.relax(psi, 3000);
    const double omega = std::sqrt(w0 * w0 + 0.25 * b * b);
    EXPECT_NEAR(prop.energy(psi), omega, 0.03 * omega);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-9);
}

TEST(PeierlsLattice2D, RelaxConfinesTheCorralGround) {
    // 1993 IBM quantum corral: 48 atoms on a ring of radius 10, B = 0. Ground
    // state lives INSIDE the ring, energy near the hard-wall J0 mode
    // j01^2 / (2 R^2) (j01 = 2.405); bracketed loosely (soft fence).
    const Grid3D g = plane_grid(16.0, 160, 16.0, 160);
    const double ring_r = 10.0;
    const double pi = std::numbers::pi;
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    for (int a = 0; a < 48; ++a) {
        const double th = 2.0 * pi * a / 48.0;
        const double ax = ring_r * std::cos(th);
        const double ay = ring_r * std::sin(th);
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double dx = g.x.coord(i) - ax;
                const double dy = g.y.coord(j) - ay;
                v[static_cast<std::size_t>(g.flat(i, j, 0))] +=
                    1.5 * std::exp(-(dx * dx + dy * dy) / (2.0 * 0.6 * 0.6));
            }
        }
    }
    ses::PeierlsLattice2D prop{g, v, 0.02};
    Field3D psi = plane_packet(g, 0.5, -1.0, 4.0, 0.0);
    prop.relax(psi, 4000);
    double inside = 0.0;
    double total = 0.0;
    for (int j = 0; j < g.y.n; ++j) {
        for (int i = 0; i < g.x.n; ++i) {
            const double w = std::norm(psi(i, j, 0));
            total += w;
            if (std::hypot(g.x.coord(i), g.y.coord(j)) < ring_r - 1.0) {
                inside += w;
            }
        }
    }
    EXPECT_GT(inside / total, 0.85);
    const double e_hard = 2.405 * 2.405 / (2.0 * ring_r * ring_r);
    const double e = prop.energy(psi);
    EXPECT_GT(e, 0.6 * e_hard);
    EXPECT_LT(e, 2.0 * e_hard);
}

// Landau-level ladder operator in the set_uniform_field gauge A = (-B y, 0):
//   a(+-) = (pi_x -+ i pi_y) / sqrt(2 B),  pi = -i grad - A;  a|n> = sqrt(n)|n-1>.
TEST(PeierlsLattice2D, LandauLadderClimbsOneCyclotronQuantum) {
    const ses::Grid3D g{ses::Grid1D{-20.0, 20.0, 128},
                        ses::Grid1D{-20.0, 20.0, 128}, ses::Grid1D{0.0, 2.0, 1}};
    const std::vector<double> zero(static_cast<std::size_t>(g.size()), 0.0);
    const double b = 0.5;
    ses::PeierlsLattice2D prop{g, zero, 0.01};
    prop.set_uniform_field(b);

    // seed near the magnetic length 1/sqrt(b) ~ 1.41
    ses::Field3D psi{g};
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            psi(i, j, 0) = std::exp(-(x * x + y * y) * 0.25);
        }
    }
    ses::normalize(psi);
    prop.relax(psi, 3000);
    ses::normalize(psi);
    const double e0 = prop.energy(psi);

    // a annihilates the LLL (residual = lattice artifact).
    ses::Field3D down = ses::landau_ladder(psi, b, false);
    EXPECT_LT(ses::norm_sq(down), 0.05);

    // a-dag climbs to n = 1: <H> rises by omega_c = B (lattice band tol).
    ses::Field3D up = ses::landau_ladder(psi, b, true);
    ses::normalize(up);
    const double e1 = prop.energy(up);
    EXPECT_NEAR((e1 - e0) / b, 1.0, 0.15);

    ses::Field3D up2 = ses::landau_ladder(up, b, true);
    ses::normalize(up2);
    const double e2 = prop.energy(up2);
    EXPECT_NEAR((e2 - e1) / b, 1.0, 0.2);
}

// Single-shot drain: a normalized packet, edge absorbers, NO injection, NO
// renorm -- the norm can only fall (the electron leaving) and the screen column
// integrates the arrivals. (The scene runs the continuous-beam version.)
TEST(PeierlsLattice2D, SinglePacketDrainsThroughTheOpenBoundary) {
    const ses::Grid3D g{ses::Grid1D{-30.0, 30.0, 128},
                        ses::Grid1D{-15.0, 15.0, 64}, ses::Grid1D{0.0, 2.0, 1}};
    const std::vector<double> zero(static_cast<std::size_t>(g.size()), 0.0);
    const double dt = 0.01;
    ses::PeierlsLattice2D prop{g, zero, dt};
    // Quadratic CAP W = W0 (1 - d/width)^2, absorb exp(-W dt) per step: the
    // cos^2 display mask is too stiff here (reflects ~30% of a slow k0 = 1
    // packet back in).
    const double w0 = 4.0;
    const double width = 6.0;
    std::vector<double> mask(static_cast<std::size_t>(g.size()));
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        const double dy = std::min(y - g.y.xmin, g.y.xmax - y);
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            const double dx = std::min(x - g.x.xmin, g.x.xmax - x);
            double w = 0.0;
            if (dx < width) {
                const double t = 1.0 - dx / width;
                w += w0 * t * t;
            }
            if (dy < width) {
                const double t = 1.0 - dy / width;
                w += w0 * t * t;
            }
            mask[static_cast<std::size_t>(g.flat(i, j, 0))] =
                std::exp(-w * dt);
        }
    }
    const double k0 = 1.0;
    ses::Field3D psi{g};
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            const double dxs = x + 18.0;
            psi(i, j, 0) = std::exp(-dxs * dxs / 36.0 - y * y / 50.0) *
                           std::complex<double>{std::cos(k0 * x),
                                                std::sin(k0 * x)};
        }
    }
    const double n0 = ses::norm_sq(psi);
    ASSERT_GT(n0, 0.0);
    for (auto& c : psi.data()) {
        c /= std::sqrt(n0);
    }
    const int i_scr = 102;  // x ~ +18, the screen column
    double arrivals = 0.0;
    double arrivals_early = 0.0;
    auto run = [&](int nsteps) {
        for (int s = 0; s < nsteps; ++s) {
            prop.step(psi, 1);
            for (std::size_t i = 0; i < psi.data().size(); ++i) {
                psi.data()[i] *= mask[i];
            }
            for (int j = 0; j < g.y.n; ++j) {
                arrivals += std::norm(psi(i_scr, j, 0)) * dt;
            }
        }
    };
    // v_g = sin(k0 h)/h ~ 0.96: launch -18 -> screen +18 -> +x absorber.
    run(1000);
    arrivals_early = arrivals;
    const double n_mid = ses::norm_sq(psi);
    run(5000);
    const double n_late = ses::norm_sq(psi);
    EXPECT_LE(n_mid, 1.0 + 1e-9);
    EXPECT_LE(n_late, n_mid + 1e-9);
    EXPECT_LT(n_late, 0.1);
    EXPECT_GT(arrivals, 100.0 * std::max(arrivals_early, 1e-30));
    EXPECT_GT(arrivals, 1e-4);
}

// The central-difference ladder explodes near the lattice band top, so the
// scene guards each jump with a measurement-based rung check (the 1D ladder_cap
// rule): <H> must move by omega_c = B within 15%, else REFUSED.
TEST(Landau2DDirector, LadderRefusesPastTheLatticeBand) {
    ses_shell::Landau2DDirector d;
    ses_shell::LandauApi* api = d.landau();
    ASSERT_NE(api, nullptr);
    int climbed = 0;
    while (climbed < 40 && api->ladder(true)) {
        ++climbed;
    }
    // The band ceiling scales as h^-2, so the rung count tracks the grid:
    // never immediately, never unbounded.
    EXPECT_GE(climbed, 15);
    EXPECT_LE(climbed, 35);
    // the Landau index followed the rungs
    const double top_n = api->mean_n();
    EXPECT_GT(top_n, 0.6 * climbed);
    // Descending is refused on these coherent-displaced states: a|alpha> ~
    // alpha|alpha> removes no clean quantum, the energy drop is ~0, the floor
    // guard strikes at once. Contract = TERMINATION + monotonicity, not unwinding.
    int descended = 0;
    while (descended < 40 && api->ladder(false)) {
        ++descended;
    }
    EXPECT_LT(descended, 40);
    EXPECT_LE(api->mean_n(), top_n + 1e-9);
}

}  // namespace
