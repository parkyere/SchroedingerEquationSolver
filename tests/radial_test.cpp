// RED: radial engine -- E1 lifetimes for bound orbitals up to n.
// 3D grid can't hold Rydberg states (n=10 ~ 400 Bohr); spherical V reduces the
// eigenproblem exactly to 1D per l.


#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>
import ses.radial;
import ses.decay;

namespace {

using ses::RadialGrid;

std::vector<double> zero_potential(const RadialGrid& g) {
    return std::vector<double>(static_cast<std::size_t>(g.n), 0.0);
}

std::vector<double> harmonic_potential_r(const RadialGrid& g) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        v[static_cast<std::size_t>(i)] = 0.5 * g.r(i) * g.r(i);
    }
    return v;
}

std::vector<double> coulomb_potential_r(const RadialGrid& g) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        v[static_cast<std::size_t>(i)] = -1.0 / g.r(i);
    }
    return v;
}

std::vector<double> soft_coulomb_potential_r(const RadialGrid& g, double a) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        v[static_cast<std::size_t>(i)] = -1.0 / std::sqrt(g.r(i) * g.r(i) + a * a);
    }
    return v;
}

int count_interior_sign_changes(const std::vector<double>& u) {
    int changes = 0;
    double prev = 0.0;
    for (const double x : u) {
        if (x != 0.0) {
            if (prev != 0.0 && x * prev < 0.0) {
                ++changes;
            }
            prev = x;
        }
    }
    return changes;
}

constexpr double kAuPerSecond = 1.0 / 2.4188843265857e-17;  // au of time in 1 s

TEST(SturmCount, MatchesTheExactDiscreteBoxSpectrum) {
    const RadialGrid g{1.0, 499};
    const ses::RadialHamiltonian h = ses::radial_hamiltonian(g, zero_potential(g), 0);
    const double dh = g.h();
    // Exact FD box spectrum of -(1/2)d2/dr2, u(0)=u(R)=0.
    auto exact = [&](int k) {
        return (1.0 - std::cos(k * 3.14159265358979323846 / (g.n + 1))) / (dh * dh);
    };
    EXPECT_EQ(ses::sturm_count_below(h, 0.5 * exact(1)), 0);
    EXPECT_EQ(ses::sturm_count_below(h, 0.5 * (exact(1) + exact(2))), 1);
    EXPECT_EQ(ses::sturm_count_below(h, 0.5 * (exact(3) + exact(4))), 3);
    EXPECT_EQ(ses::sturm_count_below(h, exact(g.n) + 1.0), g.n);
}

TEST(RadialEigenstate, BoxEigenvaluesExactAndNodesCounted) {
    const RadialGrid g{1.0, 499};
    const ses::RadialHamiltonian h = ses::radial_hamiltonian(g, zero_potential(g), 0);
    const double dh = g.h();
    for (int k = 0; k < 3; ++k) {
        const ses::RadialState s = ses::radial_eigenstate(g, h, k);
        const double exact =
            (1.0 - std::cos((k + 1) * 3.14159265358979323846 / (g.n + 1))) / (dh * dh);
        EXPECT_NEAR(s.energy, exact, 1e-9 * exact);
        EXPECT_EQ(count_interior_sign_changes(s.u), k);
        double norm = 0.0;
        for (const double x : s.u) {
            norm += x * x;
        }
        EXPECT_NEAR(norm * dh, 1.0, 1e-10);
    }
}

TEST(RadialEigenstate, IsotropicHarmonicLadder) {
    const RadialGrid g{12.0, 2399};
    const std::vector<double> v = harmonic_potential_r(g);
    // E = w (2k + l + 3/2), w = 1.
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 0), 0).energy,
                1.5, 1e-3);
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 1), 0).energy,
                2.5, 1e-3);
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 2), 0).energy,
                3.5, 1e-3);
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 0), 1).energy,
                3.5, 1e-3);
}

TEST(RadialEigenstate, HydrogenRydbergSeries) {
    const RadialGrid g{200.0, 9999};
    const std::vector<double> v = coulomb_potential_r(g);
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 0), 0).energy,
                -0.5, 1e-3);
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 0), 1).energy,
                -0.125, 5e-4);
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 1), 0).energy,
                -0.125, 5e-4);
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 2), 0).energy,
                -1.0 / 18.0, 5e-4);
    // n=5, l=4: -1/(2*25) = -0.02.
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 4), 0).energy,
                -0.02, 2e-4);
}

TEST(RadialDipole, HydrogenAnalyticIntegral) {
    const RadialGrid g{200.0, 9999};
    const std::vector<double> v = coulomb_potential_r(g);
    const ses::RadialState u10 =
        ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 0), 0);
    const ses::RadialState u21 =
        ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 1), 0);
    const double rint = ses::radial_dipole_integral(g, u10.u, u21.u);
    // <u_10|r|u_21> = 128 sqrt(6)/243.
    EXPECT_NEAR(std::abs(rint), 128.0 * std::sqrt(6.0) / 243.0, 2e-3);
}

TEST(EinsteinALevel, AngularFactorsAreExact) {
    const double alpha3 = std::pow(ses::kFineStructureConstant, 3.0);
    // l=1->0: max(1,0)/(2*1+1) = 1/3.
    EXPECT_DOUBLE_EQ(ses::einstein_a_level(1.0, 1, 0, 1.0), (4.0 / 9.0) * alpha3);
    // l=0->1: max(0,1)/(2*0+1) = 1.
    EXPECT_DOUBLE_EQ(ses::einstein_a_level(1.0, 0, 1, 1.0), (4.0 / 3.0) * alpha3);
    // l=2->1: max(2,1)/(2*2+1) = 2/5.
    EXPECT_DOUBLE_EQ(ses::einstein_a_level(1.0, 2, 1, 1.0),
                     (4.0 / 3.0) * alpha3 * (2.0 / 5.0));
}

TEST(BoundLevelTable, HydrogenLifetimesMatchTheMeasuredValues) {
    const RadialGrid g{200.0, 9999};
    const std::vector<ses::LevelInfo> table =
        ses::bound_level_table(g, coulomb_potential_r(g), 3);
    ASSERT_EQ(table.size(), 6u);  // 1s 2s 2p 3s 3p 3d

    auto level = [&](int n, int l) -> const ses::LevelInfo& {
        for (const ses::LevelInfo& e : table) {
            if (e.n == n && e.l == l) {
                return e;
            }
        }
        static const ses::LevelInfo missing{};
        ADD_FAILURE() << "missing level n=" << n << " l=" << l;
        return missing;
    };

    EXPECT_EQ(level(1, 0).lifetime, 0.0);  // stable (0 = no open E1 channel)
    EXPECT_EQ(level(2, 0).lifetime, 0.0);  // 2s: E1-stable (two-photon in QED)
    // Measured hydrogen lifetimes (au).
    EXPECT_NEAR(level(2, 1).lifetime, 1.60e-9 * kAuPerSecond,
                0.03 * 1.60e-9 * kAuPerSecond);
    EXPECT_NEAR(level(3, 0).lifetime, 158.0e-9 * kAuPerSecond,
                0.05 * 158.0e-9 * kAuPerSecond);
    EXPECT_NEAR(level(3, 1).lifetime, 5.4e-9 * kAuPerSecond,
                0.05 * 5.4e-9 * kAuPerSecond);
    EXPECT_NEAR(level(3, 2).lifetime, 15.6e-9 * kAuPerSecond,
                0.05 * 15.6e-9 * kAuPerSecond);
}

TEST(BoundLevelTable, CountsAllFiftyFiveLevelsUpToNTen) {
    const RadialGrid g{600.0, 14999};
    const std::vector<ses::LevelInfo> table =
        ses::bound_level_table(g, coulomb_potential_r(g), 10);
    ASSERT_EQ(table.size(), 55u);
    for (const ses::LevelInfo& e : table) {
        if (e.n == 1 || (e.n == 2 && e.l == 0)) {
            EXPECT_EQ(e.lifetime, 0.0);
        } else if (e.n > 1) {
            EXPECT_GT(e.lifetime, 0.0);
        }
    }
    auto lifetime = [&](int n, int l) {
        for (const ses::LevelInfo& e : table) {
            if (e.n == n && e.l == l) {
                return e.lifetime;
            }
        }
        return -1.0;
    };
    // circular (l=n-1) lifetimes grow steeply: tau(10,9) >> tau(3,2).
    EXPECT_GT(lifetime(10, 9), 100.0 * lifetime(3, 2));
    // n=10 Rydberg: -1/(2*100) = -0.005.
    for (const ses::LevelInfo& e : table) {
        if (e.n == 10) {
            EXPECT_NEAR(e.energy, -0.005, 2e-4);
        }
    }
}

TEST(BoundLevelTable, SoftCoreAtomMatchesThe3DSolver) {
    const RadialGrid g{200.0, 9999};
    const std::vector<ses::LevelInfo> table =
        ses::bound_level_table(g, soft_coulomb_potential_r(g, 1.0), 2);
    auto level = [&](int n, int l) {
        for (const ses::LevelInfo& e : table) {
            if (e.n == n && e.l == l) {
                return e;
            }
        }
        return ses::LevelInfo{};
    };
    // Reference: 3D ITP energies from the 128^3 GPU solver.
    EXPECT_NEAR(level(1, 0).energy, -0.2749, 2e-3);
    EXPECT_NEAR(level(2, 1).energy, -0.1129, 2e-3);
    EXPECT_NEAR(level(2, 0).energy, -0.0927, 2e-3);
    // 3D pipeline oracle: tau(2p_z) = 1.93e8 au (independent discretization, same physics).
    EXPECT_NEAR(level(2, 1).lifetime, 1.93e8, 0.05 * 1.93e8);
}

}  // namespace
