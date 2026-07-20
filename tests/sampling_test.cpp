// RED: trilinear complex-field sampling + per-vertex mesh phase colors.

#include <complex>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>
import ses.sampling;
import ses.grid;
import ses.vec;
import ses.marching_cubes;
import ses.field;
import ses.colormap;


namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Mesh;
using ses::Rgb;
using ses::Vec3d;

Grid3D integer_grid() {
    const Grid1D axis{0.0, 8.0, 8};  // coords 0..7 exactly
    return Grid3D{axis, axis, axis};
}

TEST(TrilinearSample, ExactAtGridPoints) {
    Field3D f{integer_grid()};
    f(2, 3, 4) = std::complex<double>{1.5, -2.5};
    const std::complex<double> s = ses::sample_trilinear(f, Vec3d{2.0, 3.0, 4.0});
    EXPECT_EQ(s.real(), 1.5);
    EXPECT_EQ(s.imag(), -2.5);
}

TEST(TrilinearSample, MidpointOfEdgeAveragesEndpoints) {
    Field3D f{integer_grid()};
    f(2, 3, 4) = std::complex<double>{1.0, 4.0};
    f(3, 3, 4) = std::complex<double>{3.0, -2.0};
    const std::complex<double> s = ses::sample_trilinear(f, Vec3d{2.5, 3.0, 4.0});
    EXPECT_DOUBLE_EQ(s.real(), 2.0);
    EXPECT_DOUBLE_EQ(s.imag(), 1.0);
}

TEST(TrilinearSample, ReproducesLinearFieldsExactly) {
    const Grid3D g = integer_grid();
    Field3D f{g};
    auto re = [](double x, double y, double z) { return 2.0 * x - 3.0 * y + z + 5.0; };
    auto im = [](double x, double y, double z) { return -x + 0.5 * y + 2.0 * z - 1.0; };
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i), y = g.y.coord(j), z = g.z.coord(k);
                f(i, j, k) = std::complex<double>{re(x, y, z), im(x, y, z)};
            }
        }
    }
    const Vec3d probes[] = {{1.3, 4.7, 2.2}, {0.1, 0.9, 6.3}, {5.999, 3.5, 0.5}};
    for (const Vec3d& p : probes) {
        const std::complex<double> s = ses::sample_trilinear(f, p);
        EXPECT_NEAR(s.real(), re(p.x, p.y, p.z), 1e-12);
        EXPECT_NEAR(s.imag(), im(p.x, p.y, p.z), 1e-12);
    }
}

TEST(TrilinearSample, WorksAtTheLastGridPoint) {
    Field3D f{integer_grid()};
    f(7, 7, 7) = std::complex<double>{9.0, -9.0};
    const std::complex<double> s = ses::sample_trilinear(f, Vec3d{7.0, 7.0, 7.0});
    EXPECT_DOUBLE_EQ(s.real(), 9.0);
    EXPECT_DOUBLE_EQ(s.imag(), -9.0);
}

TEST(PhaseColors, UniformPhaseGivesExactlyUniformColor) {
    // A(r) e^{i phi0}: interp scales amplitude only -> phase = phi0 exact.
    const Grid1D axis{-8.0, 8.0, 16};
    const Grid3D g{axis, axis, axis};
    Field3D psi{g};
    const double phi0 = 0.7;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i), y = g.y.coord(j), z = g.z.coord(k);
                const double a = std::exp(-(x * x + y * y + z * z) / 8.0);
                psi(i, j, k) = std::complex<double>{a * std::cos(phi0), a * std::sin(phi0)};
            }
        }
    }
    const Mesh mesh =
        ses::marching_cubes_at_fraction(ses::probability_density(psi), g, 0.3);
    ASSERT_GT(mesh.vertices.size(), 0u);

    const std::vector<Rgb> colors = ses::phase_colors(mesh, psi);
    ASSERT_EQ(colors.size(), mesh.vertices.size());
    const Rgb expected = ses::phase_color(phi0);
    for (const Rgb& c : colors) {
        EXPECT_NEAR(c.r, expected.r, 1e-9);
        EXPECT_NEAR(c.g, expected.g, 1e-9);
        EXPECT_NEAR(c.b, expected.b, 1e-9);
    }
}

TEST(PhaseColors, PlaneWavePhaseTracksKx) {
    // e^{i k x}: interpolated phase deviates from k*x by O((kh)^2) (~1e-3 rad).
    const Grid1D axis{-8.0, 8.0, 16};  // h = 1
    const Grid3D g{axis, axis, axis};
    Field3D psi{g};
    const double kx = 0.3;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                psi(i, j, k) = std::complex<double>{std::cos(kx * x), std::sin(kx * x)};
            }
        }
    }
    // hand-built probe mesh, known x; normals unused.
    Mesh mesh;
    mesh.vertices = {Vec3d{0.5, 0.25, -1.75}, Vec3d{-3.3, 2.0, 4.4}, Vec3d{5.6, -6.0, 0.0}};
    mesh.normals.resize(mesh.vertices.size());

    const std::vector<Rgb> colors = ses::phase_colors(mesh, psi);
    ASSERT_EQ(colors.size(), mesh.vertices.size());
    for (std::size_t i = 0; i < colors.size(); ++i) {
        const double phase = std::remainder(kx * mesh.vertices[i].x,
                                            2.0 * 3.14159265358979323846);
        const Rgb expected = ses::phase_color(phase);
        EXPECT_NEAR(colors[i].r, expected.r, 0.01);
        EXPECT_NEAR(colors[i].g, expected.g, 0.01);
        EXPECT_NEAR(colors[i].b, expected.b, 0.01);
    }
}

}  // namespace
