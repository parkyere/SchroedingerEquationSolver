// Loader for the offline-baked H2+ atlas (ses.h2plus_atlas_data, written by
// sesolver_genatlas). Checked in as a normal module so fixes compile here
// instead of hiding in the generator's string literal.
module;
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>
export module ses.h2plus_atlas_loader;
export import ses.h2plus_atlas_data;

export namespace ses {

// The baked exact-atlas for the R nearest to the requested one.
inline std::vector<H2plusOrbital> h2plus_atlas_baked(double R) {
    using namespace h2p_baked;
    int best = 0;
    for (int i = 1; i < kNR; ++i) {
        if (std::abs(kRgrid[i] - R) < std::abs(kRgrid[best] - R)) best = i;
    }
    std::vector<H2plusOrbital> out;
    for (int k = 0; k < kCount[best]; ++k) {
        const BakedOrb& b = kOrb[best][k];
        H2plusOrbital o;
        o.m = b.m; o.n_eta = b.n_eta; o.n_xi = b.n_xi;
        o.parity = b.parity; o.energy = b.energy; o.R = kRgrid[best];
        o.p = std::sqrt(std::max(0.0, -0.5 * b.energy * o.R * o.R));
        o.eta.resize(kProf); o.M.resize(kProf);
        o.xi.resize(kProf); o.lambda.resize(kProf);
        const double deta = 2.0 / kProf;
        const double dxi = (b.xi_max - 1.0) / kProf;
        for (int p = 0; p < kProf; ++p) {
            o.eta[static_cast<std::size_t>(p)] = -1.0 + (p + 0.5) * deta;
            o.xi[static_cast<std::size_t>(p)] = 1.0 + (p + 0.5) * dxi;
            o.M[static_cast<std::size_t>(p)] = kM[best][k][p];
            o.lambda[static_cast<std::size_t>(p)] = kLam[best][k][p];
        }
        out.push_back(std::move(o));
    }
    return out;
}

}  // namespace ses
