module;
#include <atomic>
#include <cmath>
#include <string>
#include <thread>
#include <vector>
export module ses.scenario.h2rotor_director;
export import ses.scenario.molecule_director;
import ses.rotor;
import ses.potential;
import ses.spheroidal;
import ses.observables;


// H2+ Ehrenfest rigid rotor: the nuclear axis is a classical unit vector
// driven by the electron's orientation torque; the electron is the full 3D
// TDSE in the moving two-center potential. Real nuclear mass, kicks in units
// of hbar, capped at the dissociation limit (ses::rotor_j_max from the exact
// E(R)). Same grid spacing as the H2+ scene in an 8x smaller box: a rotation
// period is watchable. CONTRACT: --selftest-rotor.


export namespace ses_shell {

constexpr double kRotBox = 20.0;  // Bohr half-extent (h = 0.3125, as H2+)
constexpr int kRotPoints = 128;
constexpr double kRotDt = 0.04;
constexpr double kRotRWant = 2.0;                // snapped to 2h like H2+
constexpr double kProtonMassAu = 1836.15267343;  // CODATA m_p / m_e
constexpr double kRotMu = 0.5 * kProtonMassAu;   // two protons
// Coriolis mixing 1s sigma_g -> 2p pi_u (gap ~0.67 Ha) stays < 1% below this.
constexpr double kRotAdiabaticOmega = 0.05;  // au
// E(R) scan for the dissociation cap (bound-well criterion).
constexpr double kRotScanRMin = 1.0;
constexpr double kRotScanRStep = 0.15;
constexpr int kRotScanSamples = 61;  // 1.0 .. 10.0 bohr

class H2RotorDirector final : public MoleculeDirectorBase, public RotorApi {
public:
    H2RotorDirector() : MoleculeDirectorBase(make()) {
        param_ = snapped_r();
        reset_rotor();
        rebuild_markers();
        // ~10 s of spheroidal solves: overlaps the GPU boot relax.
        jmax_thread_ = std::jthread([this] { jmax_.store(compute_j_max()); });
    }

    RotorApi* rotor() override { return this; }

    // ---- RotorApi ----
    bool kick(int axis, double dJ) override {
        if (!use_gpu_path() || stepping_ != BaseStepping::RealTime ||
            !prepared(0)) {
            return false;
        }
        ses::Vec3d a{};
        switch (axis) {
            case 0: a.x = 1.0; break;
            case 1: a.y = 1.0; break;
            case 2: a.z = 1.0; break;
            default: return false;
        }
        const bool ok = ses::rotor_kick(rotor_, a, dJ,
                                        static_cast<double>(j_max_blocking()));
        title_dirty_ = true;
        return ok;
    }
    ses::Vec3d axis() const override { return rotor_.n; }
    double j() const override { return ses::length(rotor_.L); }
    double omega() const override { return ses::length(ses::rotor_omega(rotor_)); }
    double period() const override { return ses::rotor_period(rotor_); }
    int j_max() const override { return jmax_.load(); }  // 0 = still computing
    // <H_el> in the CURRENT-axis potential, CPU truth (readback + FFT kinetic).
    double electronic_energy() override {
        ensure_cpu_current();
        return ses::mean_energy(sim_.psi(), current_potential());
    }

    bool handle_key(char key) override {
        switch (key) {
            case 'X': kick(0, 1.0); return true;
            case 'Y': kick(1, 1.0); return true;
            default: return MoleculeDirectorBase::handle_key(key);
        }
    }

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
        return {(-d) * rotor_.n, d * rotor_.n};
    }
    double geometry_parameter(int /*variant*/) const override {
        return snapped_r();
    }
    double clamp_parameter(double /*p*/) const override { return snapped_r(); }

    std::string title_suffix() override {
        std::string s = MoleculeDirectorBase::title_suffix();
        const int jm = jmax_.load();
        s += strf("  rotor: J = {:.1f}/{}  w = {:.4f} au  T = {:.0f} au  "
                  "adiabatic w/{:.2f} = {:.2f}  [rigid R; centrifugal stretch "
                  "ignored; grid egg-box: J drifts ~1%/quarter turn]  "
                  "X/Y = +1 hbar",
                  j(), jm > 0 ? std::to_string(jm) : std::string{"..."}, omega(),
                  period(), kRotAdiabaticOmega, omega() / kRotAdiabaticOmega);
        return s;
    }

    // Ehrenfest: after every real-time GPU batch, torque from the electron
    // -> rotor advance over the batch's dt -> potential rebuilt for the new
    // axis (half_mul reads it live on the next batch).
    void after_step_batch() override {
        if (!use_gpu_path() || stepping_ != BaseStepping::RealTime ||
            !prepared(0) || pending_gpu_steps_ <= 0) {
            return;
        }
        const double dt = pending_gpu_steps_ * sim_.dt();
        engine_.wait_async();  // the forces read psi
        const ses_vk::Engine::NuclearForces f =
            engine_.two_center_forces(snapped_r(), rotor_.n);
        const ses::Vec3d tau =
            f.ok ? ses::rotor_torque_from_forces(snapped_r(), rotor_.n, f.f1,
                                                 f.f2)
                 : ses::Vec3d{};
        ses::rotor_step(rotor_, tau, dt);
        engine_.set_two_center_potential(snapped_r(), rotor_.n);
        rebuild_markers();
    }

    void after_reset() override {
        reset_rotor();
        rebuild_markers();
    }

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

    void reset_rotor() {
        rotor_.n = ses::Vec3d{0.0, 0.0, 1.0};
        rotor_.L = ses::Vec3d{};
        rotor_.inertia = kRotMu * snapped_r() * snapped_r();
    }

    std::vector<double> current_potential() const {
        return ses::regularized_coulomb_potential(sim_.grid(), 1.0, centers());
    }

    // Bound-well cap from the exact 1s sigma_g curve E(R) + 1/R.
    static int compute_j_max() {
        std::vector<double> r;
        std::vector<double> v;
        for (int i = 0; i < kRotScanSamples; ++i) {
            const double R = kRotScanRMin + kRotScanRStep * i;
            const ses::H2plusOrbital o = ses::h2plus_orbital(R, 0, 0, 0);
            if (!o.valid) {
                continue;
            }
            r.push_back(R);
            v.push_back(o.energy + 1.0 / R);
        }
        return ses::rotor_j_max(r, v, kRotMu);
    }

    int j_max_blocking() {
        if (jmax_.load() <= 0 && jmax_thread_.joinable()) {
            jmax_thread_.join();
        }
        return jmax_.load();
    }

    void rebuild_markers() {
        const std::vector<ses::Vec3d> c = centers();
        balls_ = {ball(c[0], 0.4, 0.95f, 0.95f, 0.95f),
                  ball(c[1], 0.4, 0.95f, 0.95f, 0.95f)};
    }

    ses::RigidRotor rotor_;
    std::atomic<int> jmax_{0};
    std::jthread jmax_thread_;  // declared last: joins before jmax_ dies
};

}  // namespace ses_shell
