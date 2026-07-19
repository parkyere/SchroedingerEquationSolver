// RED: contract for ses.heightfield -- the 2D scenes' surface display.
// |psi|^2 as SURFACE HEIGHT (z = z_scale * |psi|^2 / norm), phase as
// per-vertex color (the shared ses::phase_color wheel), triangle soup
// (pos+normal per vertex, matching ses::Mesh / the mesh render path).
// stride decimates the source lattice so a 512^2 physics grid can feed
// a lighter display mesh.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>

import ses.heightfield;
import ses.colormap;
import ses.field;
import ses.grid;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;

// 4x4 plane (nz = 1), h = 0.5, coords -1, -0.5, 0, 0.5 per axis.
Grid3D small_grid() {
    return Grid3D{Grid1D{-1.0, 1.0, 4}, Grid1D{-1.0, 1.0, 4},
                  Grid1D{-1.0, 1.0, 1}};
}

TEST(Heightfield, SoupCountsAndSizes) {
    const Grid3D g = small_grid();
    Field3D psi{g};
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            psi(i, j, 0) = 1.0;
        }
    }
    const ses::Heightfield hf = ses::heightfield_surface(psi, 3.0, 2.0, 1);
    // 3x3 cells, 2 triangles each, 3 soup vertices per triangle.
    EXPECT_EQ(hf.mesh.vertices.size(), 54u);
    EXPECT_EQ(hf.mesh.normals.size(), 54u);
    EXPECT_EQ(hf.colors.size(), 54u);
}

TEST(Heightfield, HeightIsScaledDensity) {
    const Grid3D g = small_grid();
    Field3D psi{g};
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            psi(i, j, 0) = 1.0;  // |psi|^2 = 1
        }
    }
    psi(1, 2, 0) = std::complex<double>{0.0, 2.0};  // |psi|^2 = 4
    const double z_scale = 3.0;
    const double norm = 2.0;
    const ses::Heightfield hf = ses::heightfield_surface(psi, z_scale, norm, 1);
    // The first soup vertex is the (0,0) grid corner.
    EXPECT_NEAR(hf.mesh.vertices[0].x, g.x.coord(0), 1e-12);
    EXPECT_NEAR(hf.mesh.vertices[0].y, g.y.coord(0), 1e-12);
    EXPECT_NEAR(hf.mesh.vertices[0].z, z_scale * 1.0 / norm, 1e-12);
    // Every vertex sitting on the bumped sample carries its height.
    const double bumped = z_scale * 4.0 / norm;
    bool found = false;
    for (std::size_t v = 0; v < hf.mesh.vertices.size(); ++v) {
        if (std::abs(hf.mesh.vertices[v].x - g.x.coord(1)) < 1e-12 &&
            std::abs(hf.mesh.vertices[v].y - g.y.coord(2)) < 1e-12) {
            EXPECT_NEAR(hf.mesh.vertices[v].z, bumped, 1e-12);
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

TEST(Heightfield, PhaseColorsShareTheWheel) {
    const Grid3D g = small_grid();
    Field3D psi{g};
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            psi(i, j, 0) = 1.0;
        }
    }
    psi(1, 2, 0) = std::complex<double>{0.0, 2.0};  // arg = +pi/2
    const ses::Heightfield hf = ses::heightfield_surface(psi, 1.0, 1.0, 1);
    const ses::Rgb flat = ses::phase_color(0.0);
    const ses::Rgb bumped = ses::phase_color(std::atan2(2.0, 0.0));
    EXPECT_NEAR(hf.colors[0].r, flat.r, 1e-12);
    EXPECT_NEAR(hf.colors[0].g, flat.g, 1e-12);
    EXPECT_NEAR(hf.colors[0].b, flat.b, 1e-12);
    bool found = false;
    for (std::size_t v = 0; v < hf.mesh.vertices.size(); ++v) {
        if (std::abs(hf.mesh.vertices[v].x - g.x.coord(1)) < 1e-12 &&
            std::abs(hf.mesh.vertices[v].y - g.y.coord(2)) < 1e-12) {
            EXPECT_NEAR(hf.colors[v].r, bumped.r, 1e-12);
            EXPECT_NEAR(hf.colors[v].g, bumped.g, 1e-12);
            EXPECT_NEAR(hf.colors[v].b, bumped.b, 1e-12);
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

TEST(Heightfield, FlatFieldNormalsPointUp) {
    const Grid3D g = small_grid();
    Field3D psi{g};
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            psi(i, j, 0) = std::complex<double>{0.6, 0.8};  // |psi|^2 = 1
        }
    }
    const ses::Heightfield hf = ses::heightfield_surface(psi, 5.0, 1.0, 1);
    for (const ses::Vec3d& n : hf.mesh.normals) {
        EXPECT_NEAR(n.x, 0.0, 1e-12);
        EXPECT_NEAR(n.y, 0.0, 1e-12);
        EXPECT_NEAR(n.z, 1.0, 1e-12);
    }
}

TEST(Heightfield, StrideDecimatesTheLattice) {
    // 8x8 plane, stride 2 -> samples 0,2,4,6 -> 4x4 vertex grid -> 54 soup.
    const Grid3D g{Grid1D{-2.0, 2.0, 8}, Grid1D{-2.0, 2.0, 8},
                   Grid1D{-1.0, 1.0, 1}};
    Field3D psi{g};
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            psi(i, j, 0) = 1.0;
        }
    }
    const ses::Heightfield hf = ses::heightfield_surface(psi, 1.0, 1.0, 2);
    EXPECT_EQ(hf.mesh.vertices.size(), 54u);
    // Decimated vertices sit on the SOURCE lattice coordinates.
    EXPECT_NEAR(hf.mesh.vertices[0].x, g.x.coord(0), 1e-12);
    bool has_stride_point = false;
    for (const ses::Vec3d& v : hf.mesh.vertices) {
        if (std::abs(v.x - g.x.coord(2)) < 1e-12) {
            has_stride_point = true;
        }
    }
    EXPECT_TRUE(has_stride_point);
}

TEST(Heightfield, NonPositiveNormMeansFlatZero) {
    const Grid3D g = small_grid();
    Field3D psi{g};
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            psi(i, j, 0) = 7.0;
        }
    }
    const ses::Heightfield hf = ses::heightfield_surface(psi, 3.0, 0.0, 1);
    for (const ses::Vec3d& v : hf.mesh.vertices) {
        EXPECT_EQ(v.z, 0.0);
    }
}

}  // namespace
