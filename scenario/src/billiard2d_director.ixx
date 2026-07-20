module;
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.billiard2d_director;
export import ses.scenario.lattice2d_director;
import ses.heightfield;
import ses.observables;
import ses.propagator;
import ses.parallel;


// Quantum billiard: one hard 2D table, two geometries. The CIRCLE is
// integrable -- a tangential packet conserves |L|, so its orbit never
// enters the caustic disk and the TIME-AVERAGED density keeps a dark
// hole. The Bunimovich STADIUM (same table with the caps pulled apart by
// a straight section) is fully chaotic -- the flat walls break L, orbits
// visit everywhere, and the average fills the center; along short
// periodic orbits the average keeps faint SCARS. Spectral split-operator
// on (512, 512, 1) (smooth quadratic wall, B = 0); the average view is
// the scar lens. CONTRACT: tests/billiard2d_test.cpp.


export namespace ses_shell {

// Signed distance to the stadium boundary: the stadium is every point
// within `r` of the segment y = 0, |x| <= half_len (half_len = 0 is the
// circle). Negative inside.
inline double stadium_sdf(double /*x*/, double /*y*/, double /*half_len*/,
                          double /*r*/) {
    return 0.0;  // RED stub
}

}  // namespace ses_shell
