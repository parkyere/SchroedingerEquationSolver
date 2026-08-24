module;
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>
export module ses.imaginary_time;
export import ses.grid;
export import ses.spectral;
export import ses.fft;
export import ses.field;
import ses.parallel;


// Imaginary-time relaxation e^{-H dtau}; Strang split, non-unitary -> renormalize each step.


export namespace ses {

namespace itp_detail {

inline double exp_weight(double th) noexcept { return std::exp(th); }

}  // namespace itp_detail

class ImaginaryTimePropagator1D {
public:
    ImaginaryTimePropagator1D(const Grid1D& g, const std::vector<double>& potential,
                              double dtau)
        : half_v_(build_half_potential_table(potential, dtau, itp_detail::exp_weight)),
          kinetic_(build_kinetic_table(g, 1.0, dtau, itp_detail::exp_weight)) {
        assert(static_cast<int>(potential.size()) == g.n);
    }

    void relax(Field1D& psi, int nsteps) const {
        assert(psi.data().size() == half_v_.size());
        for (int s = 0; s < nsteps; ++s) {
            apply_weight(half_v_, psi.data());
            fft(psi.data());
            apply_weight(kinetic_, psi.data());
            ifft(psi.data());
            apply_weight(half_v_, psi.data());
            normalize(psi);
        }
    }

private:
    static void apply_weight(const std::vector<double>& weight,
                             std::vector<std::complex<double>>& a) noexcept {
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = weight[i] * a[i];
        }
    }

    std::vector<double> half_v_;
    std::vector<double> kinetic_;
};

class ImaginaryTimePropagator3D {
public:
    // mass default 1.0: bitwise-identical to legacy tables (MassParameter tests).
    ImaginaryTimePropagator3D(const Grid3D& g, const std::vector<double>& potential,
                              double dtau, double mass = 1.0)
        : half_v_(build_half_potential_table(potential, dtau, itp_detail::exp_weight)),
          kinetic_(build_kinetic_table(g, mass, dtau, itp_detail::exp_weight)) {
        assert(static_cast<int>(potential.size()) == g.size());
    }

    // GPU relax path consumes these TESTED tables instead of re-deriving.
    const std::vector<double>& half_potential_weight() const noexcept { return half_v_; }
    const std::vector<double>& kinetic_weight() const noexcept { return kinetic_; }

    void relax(Field3D& psi, int nsteps) const { relax_deflated(psi, {}, nsteps); }

    // Gram-Schmidt deflation of `lower` each step -> converges to the next excited state.
    void relax_deflated(Field3D& psi, const std::vector<const Field3D*>& lower,
                        int nsteps) const {
        assert(psi.data().size() == half_v_.size());
        for (int s = 0; s < nsteps; ++s) {
            apply_weight(half_v_, psi.data());
            fft(psi);
            apply_weight(kinetic_, psi.data());
            ifft(psi);
            apply_weight(half_v_, psi.data());
            for (const Field3D* phi : lower) {
                const std::complex<double> c = inner_product(*phi, psi);
                std::vector<std::complex<double>>& p = psi.data();
                const std::vector<std::complex<double>>& q = phi->data();
                for (std::size_t i = 0; i < p.size(); ++i) {
                    p[i] = p[i] - c * q[i];
                }
            }
            normalize(psi);
        }
    }

private:
    // Elementwise (disjoint) scale: threaded result is bitwise identical.
    static void apply_weight(const std::vector<double>& weight,
                             std::vector<std::complex<double>>& a) noexcept {
        parallel_for(static_cast<int>(a.size()), [&](int i) {
            a[static_cast<std::size_t>(i)] =
                weight[static_cast<std::size_t>(i)] * a[static_cast<std::size_t>(i)];
        });
    }

    std::vector<double> half_v_;
    std::vector<double> kinetic_;
};

}  // namespace ses
