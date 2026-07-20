// Periodic grid: h = L/n (NOT endpoint-inclusive L/(n-1)); xmax aliases to xmin.
// Endpoint-inclusive spacing breaks the split-operator FFT wavenumber mapping.

#include <gtest/gtest.h>
import ses.grid;

namespace {

using ses::Grid1D;

TEST(Grid1D, SizeIsPointCount) {
    constexpr Grid1D g{0.0, 10.0, 10};
    EXPECT_EQ(g.size(), 10);
}

TEST(Grid1D, SpacingIsExtentOverN) {
    constexpr Grid1D g{0.0, 10.0, 10};
    EXPECT_EQ(g.spacing(), 1.0);
}

TEST(Grid1D, CoordOfFirstPointIsXmin) {
    constexpr Grid1D g{-5.0, 5.0, 100};
    EXPECT_EQ(g.coord(0), -5.0);
}

TEST(Grid1D, LastPointStopsOneStepShortOfXmax) {
    constexpr Grid1D g{0.0, 10.0, 10};
    EXPECT_EQ(g.coord(9), 9.0);
}

TEST(Grid1D, CenterPointOfSymmetricGrid) {
    constexpr Grid1D g{-5.0, 5.0, 100};
    EXPECT_DOUBLE_EQ(g.coord(50), 0.0);
}

TEST(Grid1D, ExtentAccessors) {
    constexpr Grid1D g{-2.0, 3.0, 50};
    EXPECT_EQ(g.xmin, -2.0);
    EXPECT_EQ(g.xmax, 3.0);
}

}  // namespace
