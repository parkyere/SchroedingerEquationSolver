// RED: phasor curve = (x_i, r cos phi, r sin phi), r = scale |psi_i|^2, phi = arg psi_i.
// potential = second polyline in z=0, y = min(v_i * e_scale, y_clamp).

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

import ses.colormap;
import ses.field;
import ses.grid;
import ses.phasor;

namespace {

const ses::Grid1D kGrid{0.0, 8.0, 8};

TEST(PhasorCurve, EmitsOneVertexPerGridPointWithXAscending) {
    ses::Field1D psi{kGrid};
    const std::vector<float> v = ses::phasor_curve(psi, 1.0);
    ASSERT_EQ(v.size(), static_cast<std::size_t>(3 * kGrid.n));
    for (int i = 0; i < kGrid.n; ++i) {
        EXPECT_FLOAT_EQ(v[static_cast<std::size_t>(3 * i)],
                        static_cast<float>(kGrid.coord(i)));
    }
}

TEST(PhasorCurve, RadiusIsAmplitudeSquaredAndAngleIsThePhase) {
    ses::Field1D psi{kGrid};
    psi[1] = std::complex<double>{2.0, 0.0};
    psi[2] = std::complex<double>{0.0, 2.0};
    psi[3] = std::complex<double>{-3.0, 0.0};
    psi[4] = std::complex<double>{0.0, -1.0};
    const double s = 0.5;
    const std::vector<float> v = ses::phasor_curve(psi, s);
    auto y = [&](int i) { return v[static_cast<std::size_t>(3 * i + 1)]; };
    auto z = [&](int i) { return v[static_cast<std::size_t>(3 * i + 2)]; };
    EXPECT_NEAR(y(1), s * 4.0, 1e-5);
    EXPECT_NEAR(z(1), 0.0, 1e-5);
    EXPECT_NEAR(y(2), 0.0, 1e-5);
    EXPECT_NEAR(z(2), s * 4.0, 1e-5);
    EXPECT_NEAR(y(3), -s * 9.0, 1e-5);
    EXPECT_NEAR(z(3), 0.0, 1e-5);
    EXPECT_NEAR(y(4), 0.0, 1e-5);
    EXPECT_NEAR(z(4), -s * 1.0, 1e-5);
}

TEST(PhasorCurve, VanishingAmplitudeSitsOnTheAxis) {
    ses::Field1D psi{kGrid};
    const std::vector<float> v = ses::phasor_curve(psi, 3.0);
    for (int i = 0; i < kGrid.n; ++i) {
        EXPECT_EQ(v[static_cast<std::size_t>(3 * i + 1)], 0.0f);
        EXPECT_EQ(v[static_cast<std::size_t>(3 * i + 2)], 0.0f);
    }
}

TEST(PhasorCurve, GlobalPhaseRotatesTheWholeCurveRigidly) {
    // exp(i alpha) psi rotates every vertex by alpha about the x axis.
    ses::Field1D psi{kGrid};
    for (int i = 0; i < kGrid.n; ++i) {
        psi[i] = std::complex<double>{0.3 + 0.1 * i, 0.2};
    }
    const double alpha = 0.7;
    ses::Field1D rotated = psi;
    const std::complex<double> ph{std::cos(alpha), std::sin(alpha)};
    for (int i = 0; i < kGrid.n; ++i) {
        rotated[i] *= ph;
    }
    const std::vector<float> a = ses::phasor_curve(psi, 2.0);
    const std::vector<float> b = ses::phasor_curve(rotated, 2.0);
    const double c = std::cos(alpha);
    const double sn = std::sin(alpha);
    for (int i = 0; i < kGrid.n; ++i) {
        const double ya = a[static_cast<std::size_t>(3 * i + 1)];
        const double za = a[static_cast<std::size_t>(3 * i + 2)];
        EXPECT_NEAR(b[static_cast<std::size_t>(3 * i + 1)], c * ya - sn * za, 1e-5);
        EXPECT_NEAR(b[static_cast<std::size_t>(3 * i + 2)], sn * ya + c * za, 1e-5);
    }
}

// RED: density band = |psi|^2 shadow on z=0, TRIANGLE_STRIP. Colors use the same
// phase_color wheel as the 3D volume view (hue is scene-invariant), premultiplied
// for the overlay blend.
TEST(DensityBand, ProjectsAmplitudeSquaredSymmetricallyOntoThePlane) {
    ses::Field1D psi{kGrid};
    psi[1] = std::complex<double>{2.0, 0.0};
    psi[3] = std::complex<double>{0.0, -1.5};
    const double s = 0.5;
    const std::vector<float> b = ses::density_band(psi, s);
    ASSERT_EQ(b.size(), static_cast<std::size_t>(6 * kGrid.n));
    for (int i = 0; i < kGrid.n; ++i) {
        const std::size_t o = static_cast<std::size_t>(6 * i);
        const double r = s * std::norm(psi[i]);
        EXPECT_FLOAT_EQ(b[o + 0], static_cast<float>(kGrid.coord(i)));
        EXPECT_NEAR(b[o + 1], -r, 1e-6);
        EXPECT_EQ(b[o + 2], 0.0f);
        EXPECT_FLOAT_EQ(b[o + 3], static_cast<float>(kGrid.coord(i)));
        EXPECT_NEAR(b[o + 4], r, 1e-6);
        EXPECT_EQ(b[o + 5], 0.0f);
    }
}

TEST(DensityBand, ColorsCarryTheVolumePhaseWheelPremultiplied) {
    ses::Field1D psi{kGrid};
    psi[1] = std::complex<double>{2.0, 0.0};
    psi[2] = std::complex<double>{0.0, 2.0};
    psi[3] = std::complex<double>{-3.0, 0.0};
    psi[4] = std::complex<double>{1.0, -1.0};
    const float alpha = 0.35f;
    const std::vector<float> c = ses::phase_band_colors(psi, alpha);
    ASSERT_EQ(c.size(), static_cast<std::size_t>(8 * kGrid.n));
    for (int i : {1, 2, 3, 4}) {
        const ses::Rgb want = ses::phase_color(std::arg(psi[i]));
        for (int v = 0; v < 2; ++v) {
            const std::size_t o = static_cast<std::size_t>(8 * i + 4 * v);
            EXPECT_NEAR(c[o + 0], want.r * alpha, 1e-6) << "i=" << i;
            EXPECT_NEAR(c[o + 1], want.g * alpha, 1e-6) << "i=" << i;
            EXPECT_NEAR(c[o + 2], want.b * alpha, 1e-6) << "i=" << i;
            EXPECT_NEAR(c[o + 3], alpha, 1e-6) << "i=" << i;
        }
    }
}

TEST(PotentialCurve, MapsToTheYPlaneAndClampsTheWalls) {
    std::vector<double> v(8, 0.0);
    v[2] = 1.0;
    v[5] = 40.0;
    const std::vector<float> c = ses::potential_curve(kGrid, v, 0.5, 4.0);
    ASSERT_EQ(c.size(), static_cast<std::size_t>(3 * kGrid.n));
    for (int i = 0; i < kGrid.n; ++i) {
        EXPECT_FLOAT_EQ(c[static_cast<std::size_t>(3 * i)],
                        static_cast<float>(kGrid.coord(i)));
        EXPECT_EQ(c[static_cast<std::size_t>(3 * i + 2)], 0.0f);
    }
    EXPECT_NEAR(c[3 * 2 + 1], 0.5, 1e-6);
    EXPECT_NEAR(c[3 * 5 + 1], 4.0, 1e-6);  // 40 * 0.5 = 20 -> clamped
    EXPECT_EQ(c[3 * 0 + 1], 0.0f);
}

}  // namespace
