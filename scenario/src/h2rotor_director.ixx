module;
#include <cmath>
#include <string>
#include <vector>
export module ses.scenario.h2rotor_director;
export import ses.scenario.molecule_director;
import ses.rotor;
import ses.potential;


// H2+ Ehrenfest rigid rotor: the nuclear axis is a classical unit vector
// driven by the electron's orientation torque; the electron is the full 3D
// TDSE in the moving two-center potential. Real nuclear mass, kicks in units
// of hbar, capped at the dissociation limit (ses::rotor_j_max). Same grid
// spacing as the H2+ scene in an 8x smaller box: a rotation period is
// watchable. CONTRACT: --selftest-rotor.


export namespace ses_shell {

constexpr double kRotBox = 20.0;  // Bohr half-extent (h = 0.3125, as H2+)
constexpr int kRotPoints = 128;
constexpr double kRotDt = 0.04;
constexpr double kRotRWant = 2.0;             // snapped to 2h like the H2+ scene
constexpr double kProtonMassAu = 1836.15267343;  // CODATA m_p / m_e
constexpr double kRotMu = 0.5 * kProtonMassAu;   // reduced mass of two protons

class H2RotorDirector final : public MoleculeDirectorBase, public RotorApi {
public:
    H2RotorDirector() : MoleculeDirectorBase(make()) { param_ = snapped_r(); }

    RotorApi* rotor() override { return this; }

    // ---- RotorApi (RED stubs: --selftest-rotor must fail before wiring) ----
    bool kick(int /*axis*/, double /*dJ*/) override { return false; }
    ses::Vec3d axis() const override { return ses::Vec3d{0.0, 0.0, 1.0}; }
    double j() const override { return 0.0; }
    double omega() const override { return 0.0; }
    double period() const override { return 0.0; }
    int j_max() const override { return 0; }
    double electronic_energy() override { return 0.0; }

    // R is rigid: no geometry knobs.
    void set_geometry(int /*variant*/) override {}
    void set_parameter(double /*p*/) override {}

    double default_camera_azimuth() const override { return 0.35; }
    double default_camera_elevation() const override { return 0.28; }
    double default_camera_distance() const override { return 50.0; }

protected:
    const char* scene_name() const override { return "H2+ rotor (Ehrenfest)"; }
    ses::WavepacketSimulation remake_simulation() const override { return make(); }
    int exposed_states() const override { return 1; }
    const char* orbital_name(int /*k*/) const override { return "1s sigma_g"; }
    std::vector<ses::Vec3d> centers() const override {
        const double d = 0.5 * snapped_r();
        return {{0.0, 0.0, -d}, {0.0, 0.0, d}};
    }
    double geometry_parameter(int /*variant*/) const override {
        return snapped_r();
    }
    double clamp_parameter(double /*p*/) const override { return snapped_r(); }

private:
    // Nuclei on-grid at boot (R in multiples of 2h, as the H2+ scene snaps).
    static double snapped_r() {
        const double h = 2.0 * kRotBox / kRotPoints;
        return 2.0 * h * std::max(1.0, std::round(kRotRWant / (2.0 * h)));
    }

    static ses::WavepacketSimulation make() {
        const ses::Grid1D axis{-kRotBox, kRotBox, kRotPoints};
        const ses::Grid3D grid{axis, axis, axis};
        const double d = 0.5 * snapped_r();
        return ses::WavepacketSimulation{ses::WavepacketSimulation::Config{
            grid,
            ses::regularized_coulomb_potential(
                grid, 1.0, {{0.0, 0.0, -d}, {0.0, 0.0, d}}),
            ses::Vec3d{},
            ses::Vec3d{1.8, 1.8, 1.8},
            ses::Vec3d{},
            kRotDt,
        }};
    }
};

}  // namespace ses_shell
