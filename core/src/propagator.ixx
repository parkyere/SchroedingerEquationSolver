module;
#include <complex>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
export module ses.propagator;
export import ses.grid;
export import ses.spectral;
export import ses.fft;
export import ses.field;


// Split-operator (Fourier) TDSE propagator, atomic units. Tables and phase
// apply shared via ses.spectral (bitwise-pinned by propagator_tables_test).


export namespace ses {

class SplitOperator1D {
public:
    SplitOperator1D(const Grid1D& g, const std::vector<double>& potential, double dt)
        : dt_(dt),
          half_v_(build_half_potential_table(potential, dt, unit_phase)),
          kinetic_(build_kinetic_table(g, 1.0, dt, unit_phase)) {
        assert(static_cast<int>(potential.size()) == g.n);
    }

    double dt() const noexcept { return dt_; }

    void step(Field1D& psi, int nsteps = 1) const {
        assert(psi.data().size() == half_v_.size());
        for (int s = 0; s < nsteps; ++s) {
            apply_phase(half_v_, psi.data());
            fft(psi.data());
            apply_phase(kinetic_, psi.data());
            ifft(psi.data());
            apply_phase(half_v_, psi.data());
        }
    }

private:
    double dt_;
    std::vector<std::complex<double>> half_v_;
    std::vector<std::complex<double>> kinetic_;
};

class SplitOperator3D {
public:
    // mass: effective m* (corral Cu(111) runs m* != 1). Default 1.0 keeps
    // existing caller tables bitwise identical (MassParameter tests).
    SplitOperator3D(const Grid3D& g, const std::vector<double>& potential, double dt,
                    double mass = 1.0)
        : dt_(dt),
          half_v_(build_half_potential_table(potential, dt, unit_phase)),
          kinetic_(build_kinetic_table(g, mass, dt, unit_phase)) {
        assert(static_cast<int>(potential.size()) == g.size());
    }

    double dt() const noexcept { return dt_; }

    // Read access for table-pinning tests; the GPU engine derives phases in-shader.
    const std::vector<std::complex<double>>& half_potential_phase() const noexcept { return half_v_; }
    const std::vector<std::complex<double>>& kinetic_phase() const noexcept { return kinetic_; }

    void step(Field3D& psi, int nsteps = 1) const {
        assert(psi.data().size() == half_v_.size());
        for (int s = 0; s < nsteps; ++s) {
            apply_phase(half_v_, psi.data());
            fft(psi);
            apply_phase(kinetic_, psi.data());
            ifft(psi);
            apply_phase(half_v_, psi.data());
        }
    }

    // The step's factors, for a TIME-DEPENDENT potential (boundary-sampled
    // Strang: half-kick V_0, [drift, full-kick V_k], drift, half-kick V_N).
    // kick(v, dt/2) is bitwise the step's own half table. CONTRACT:
    // propagator_test SplitOperator3DPrimitives, vkcheck rotating batch.
    void kick(Field3D& psi, const std::vector<double>& v, double dt) const {
        assert(psi.data().size() == v.size());
        apply_phase(build_half_potential_table(v, 2.0 * dt, unit_phase),
                    psi.data());
    }
    void drift(Field3D& psi) const {
        assert(psi.data().size() == kinetic_.size());
        fft(psi);
        apply_phase(kinetic_, psi.data());
        ifft(psi);
    }

private:
    double dt_;
    std::vector<std::complex<double>> half_v_;
    std::vector<std::complex<double>> kinetic_;
};

}  // namespace ses
