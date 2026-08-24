module;
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>
export module ses.mcwf1d;
export import ses.field;
export import ses.grid;
export import ses.imaginary_time;
import ses.ladder;
import ses.observables;


// 1D MCWF unraveling of cavity photon loss (rate kappa, jump op a).
// No-jump exactness (harmonic V): e^{-H tau} = const x e^{-n omega tau},
// so relax at dtau = kappa dt/(2 omega) reproduces e^{-kappa n dt/2}.
// CONTRACT: tests/mcwf1d_test.cpp.


export namespace ses {

// Owns the dtau = kappa dt / (2 omega) coupling and the (grid, V) binding --
// the type IS the contract; no hand-fed dtau can drift from (kappa, omega, dt).
class PhotonLossDamper {
  public:
    PhotonLossDamper(const Grid1D& g, std::vector<double> potential,
                     double omega, double kappa, double dt)
        : grid_(g),
          v_(std::move(potential)),
          omega_(omega),
          kappa_(kappa),
          dt_(dt),
          itp_(g, v_, kappa * dt / (2.0 * omega)) {}

    const Grid1D& grid() const noexcept { return grid_; }
    const std::vector<double>& potential() const noexcept { return v_; }
    double omega() const noexcept { return omega_; }
    double kappa() const noexcept { return kappa_; }
    double dt() const noexcept { return dt_; }

    void relax(Field1D& psi) const { itp_.relax(psi, 1); }

  private:
    Grid1D grid_;
    std::vector<double> v_;
    double omega_;
    double kappa_;
    double dt_;
    ImaginaryTimePropagator1D itp_;
};

// u = caller's uniform draw in [0, 1).
inline bool photon_loss_step(Field1D& psi, double u,
                             const PhotonLossDamper& damp) {
    assert(psi.grid().n == damp.grid().n &&
           psi.grid().xmin == damp.grid().xmin &&
           psi.grid().xmax == damp.grid().xmax);
    const double n_bar =
        std::max(0.0, mean_energy(psi, damp.potential()) / damp.omega() - 0.5);
    if (u < damp.kappa() * n_bar * damp.dt()) {
        ladder_lower(psi, damp.omega());  // a psi / ||.||
        return true;
    }
    damp.relax(psi);  // no-jump e^{-kappa n dt/2}/norm, EXACT (harmonic V)
    return false;
}

}  // namespace ses
