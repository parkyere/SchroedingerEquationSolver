module;
#include <algorithm>
#include <complex>
#include <cmath>
#include <cstddef>
#include <vector>
#include <cstdint>
export module ses.decay;
export import ses.grid;
export import ses.field;
import ses.parallel;


// Spontaneous decay via quantum jumps (MCWF), atomic units; caller injects randomness.


export namespace ses {

inline constexpr double kFineStructureConstant = 7.2973525693e-3;

struct DipoleMatrixElement {
    std::complex<double> x;
    std::complex<double> y;
    std::complex<double> z;
};

// Per-z-slab partials recombined in FIXED order -> deterministic run-to-run (ses.parallel).
inline DipoleMatrixElement dipole_matrix_element(const Field3D& f, const Field3D& i) {
    const Grid3D& g = f.grid();
    const int nz = g.z.n;
    std::vector<std::complex<double>> px(static_cast<std::size_t>(nz));
    std::vector<std::complex<double>> py(static_cast<std::size_t>(nz));
    std::vector<std::complex<double>> pz(static_cast<std::size_t>(nz));
    parallel_for(nz, [&](int k) {
        std::complex<double> dx{};
        std::complex<double> dy{};
        std::complex<double> dz{};
        for (int j = 0; j < g.y.n; ++j) {
            for (int ii = 0; ii < g.x.n; ++ii) {
                const std::complex<double> t = std::conj(f(ii, j, k)) * i(ii, j, k);
                dx += g.x.coord(ii) * t;
                dy += g.y.coord(j) * t;
                dz += g.z.coord(k) * t;
            }
        }
        px[static_cast<std::size_t>(k)] = dx;
        py[static_cast<std::size_t>(k)] = dy;
        pz[static_cast<std::size_t>(k)] = dz;
    });
    std::complex<double> dx{};
    std::complex<double> dy{};
    std::complex<double> dz{};
    for (int k = 0; k < nz; ++k) {
        dx += px[static_cast<std::size_t>(k)];
        dy += py[static_cast<std::size_t>(k)];
        dz += pz[static_cast<std::size_t>(k)];
    }
    const double dv = g.cell_volume();
    return DipoleMatrixElement{dv * dx, dv * dy, dv * dz};
}

inline constexpr double dipole_strength_sq(const DipoleMatrixElement& d) noexcept {
    return std::norm(d.x) + std::norm(d.y) + std::norm(d.z);
}

inline constexpr double einstein_a(double omega, double dipole_sq) noexcept {
    const double a3 = kFineStructureConstant * kFineStructureConstant *
                      kFineStructureConstant;
    return (4.0 / 3.0) * a3 * omega * omega * omega * dipole_sq;
}

struct JumpResult {
    bool jumped{};
    double p_jump{};
};

// On survival psi is untouched here; caller applies the MCWF no-jump H_eff damping separately.
inline JumpResult quantum_jump(Field3D& psi, const Field3D& excited, const Field3D& ground,
                               double gamma, double dt, double u) {
    const double p_e = std::norm(inner_product(excited, psi));
    const double p = 1.0 - std::exp(-gamma * p_e * dt);
    if (u < p) {
        psi = ground;
        return JumpResult{true, p};
    }
    return JumpResult{false, p};
}

// GPU apply_mcwf_damping mirrors this in the {|n>} basis (single source of truth).
inline std::vector<std::complex<double>> nojump_damped_amplitudes(
    const std::vector<std::complex<double>>& c, const std::vector<double>& gamma,
    double dt) {
    std::vector<std::complex<double>> out(c.size());
    double n2 = 0.0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        const double f = std::exp(-0.5 * gamma[i] * dt);
        out[i] = f * c[i];
        n2 += std::norm(out[i]);
    }
    if (n2 > 0.0) {
        const double inv = 1.0 / std::sqrt(n2);
        for (std::complex<double>& z : out) {
            z = inv * z;
        }
    }
    return out;
}

// Only wall absorption (= ionization) is real loss; add MCWF-damping known_loss
// back before ratioing. Caller: bound_survival *= this.
inline double bound_survival_ratio(double post_norm, double known_loss,
                                   double baseline) {
    return baseline > 0.0 ? (post_norm + known_loss) / baseline : 1.0;
}

// ---- multi-channel jumps ----

struct DecayChannel {
    const Field3D* from;
    const Field3D* to;
    double gamma;
};

struct MultiJumpResult {
    int channel{-1};
    double p_total{};
};

struct ChannelPick {
    int channel{-1};
    double p_total{};
};

// Pure selection over precomputed rates -- reused by the GPU shell on GPU-reduced
// populations. Final fallthrough guards u2 rounding at the top of the last stratum.
inline ChannelPick pick_decay_channel(const std::vector<double>& rates, double dt,
                                      double u1, double u2) noexcept {
    double total = 0.0;
    for (const double r : rates) {
        total += r;
    }
    if (total <= 0.0) {
        return ChannelPick{-1, 0.0};
    }
    const double p = 1.0 - std::exp(-total * dt);
    if (!(u1 < p)) {
        return ChannelPick{-1, p};
    }
    double acc = 0.0;
    int last_positive = -1;
    for (std::size_t m = 0; m < rates.size(); ++m) {
        if (rates[m] <= 0.0) {
            continue;
        }
        last_positive = static_cast<int>(m);
        acc += rates[m] / total;
        if (u2 < acc) {
            return ChannelPick{static_cast<int>(m), p};
        }
    }
    return ChannelPick{last_positive, p};
}

// ---- chained jumps (one accumulated interval) ----

struct RateChannel {
    int from;
    int to;
    double gamma;
};

// Chained first-arrival jumps across one accumulated interval: segment rates
// gamma_c * pop[from_c]; a collapse sets pop to a delta on the destination, so
// cascade hops chain within the SAME interval (single-jump trials defer them a
// whole interval; that error grows with time_scale). Draws no uniforms when no
// rate is positive; u01() supplies uniforms in [0,1).
template <class U01>
inline std::vector<int> chain_decay_jumps(const std::vector<RateChannel>& channels,
                                          std::vector<double> pop, double dt,
                                          U01&& u01) {
    std::vector<int> fired;
    std::vector<double> rates(channels.size());
    double remaining = dt;
    for (;;) {
        double total = 0.0;
        for (std::size_t m = 0; m < channels.size(); ++m) {
            rates[m] = channels[m].gamma *
                       pop[static_cast<std::size_t>(channels[m].from)];
            total += rates[m];
        }
        if (total <= 0.0) {
            break;
        }
        // t1 = -ln(u)/R; u -> 0 reads +inf and survives the interval.
        const double t1 = -std::log(u01()) / total;
        if (!(t1 < remaining)) {
            break;
        }
        const double u2 = u01();
        double acc = 0.0;
        int picked = -1;
        int last_positive = -1;
        for (std::size_t m = 0; m < channels.size(); ++m) {
            if (rates[m] <= 0.0) {
                continue;
            }
            last_positive = static_cast<int>(m);
            acc += rates[m] / total;
            if (u2 < acc) {
                picked = static_cast<int>(m);
                break;
            }
        }
        if (picked < 0) {
            picked = last_positive;  // u2 rounding at the top of the last stratum
        }
        fired.push_back(picked);
        std::fill(pop.begin(), pop.end(), 0.0);
        pop[static_cast<std::size_t>(
            channels[static_cast<std::size_t>(picked)].to)] = 1.0;
        remaining -= t1;
    }
    return fired;
}

inline MultiJumpResult multi_quantum_jump(Field3D& psi,
                                          const std::vector<DecayChannel>& channels,
                                          double dt, double u1, double u2) {
    std::vector<double> rates(channels.size());
    for (std::size_t m = 0; m < channels.size(); ++m) {
        rates[m] = channels[m].gamma * std::norm(inner_product(*channels[m].from, psi));
    }
    const ChannelPick pick = pick_decay_channel(rates, dt, u1, u2);
    if (pick.channel >= 0) {
        psi = *channels[static_cast<std::size_t>(pick.channel)].to;
    }
    return MultiJumpResult{pick.channel, pick.p_total};
}

}  // namespace ses
