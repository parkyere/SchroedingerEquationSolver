#pragma once

// ses_vk::Engine (M5 Stage 2): the framework-free analog of
// ses_qrhi::QrhiEngine's core propagation -- the split-operator Strang step
// and imaginary-time relaxation, on raw Vulkan via the vk_compute.hpp layer.
// No Qt anywhere. SPIR-V blobs are dependency-injected (EngineKernels), so
// the engine has no resource system; the DeviceContext is passed in, so the
// same engine runs on a self-created device (headless: checks, clusters) or,
// later, on handles adopted from the GUI's QRhi.
//
// Numerical contract: byte-identical to the QRhi engine's hand-rolled path.
// Same kernels (the qsb-decorated Vulkan-GLSL sources, baked offline), same
// dispatch chain (halfV . IFFT . kin . FFT . halfV; the inverse FFT = conj .
// FFT . conj/N), same std140 parameter blocks, same host-double reduction
// finishes. What QRhi did implicitly and this engine does explicitly: a
// compute-to-compute memory barrier before every dispatch that aliases psi
// (all of them), transfer barriers around uploads/readbacks, and a fence
// wait per submission (the analog of endOffscreenFrame's synchronization).

#include "vk_compute.hpp"

#include <core/complex.hpp>
#include <core/drive.hpp>
#include <core/grid.hpp>
#include <core/vec.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ses_vk {

// The SPIR-V blobs the engine's core propagation needs. The caller owns the
// storage (embedded C arrays in the harness; a file loader later).
struct EngineKernels {
    const unsigned char* mul = nullptr;    // phase_multiply.comp
    std::size_t mul_size = 0;
    const unsigned char* conj = nullptr;   // conj_scale.comp
    std::size_t conj_size = 0;
    const unsigned char* fft = nullptr;    // fft_line<n>.comp for the grid n
    std::size_t fft_size = 0;
    const unsigned char* norm = nullptr;   // norm_peak.comp
    std::size_t norm_size = 0;
    const unsigned char* scale = nullptr;  // scale.comp
    std::size_t scale_size = 0;
    const unsigned char* kick = nullptr;   // dipole_kick.comp
    std::size_t kick_size = 0;
    const unsigned char* shear = nullptr;  // shear.comp
    std::size_t shear_size = 0;
    const unsigned char* inner = nullptr;  // inner_product.comp
    std::size_t inner_size = 0;
    const unsigned char* axpy = nullptr;   // axpy.comp
    std::size_t axpy_size = 0;
    const unsigned char* copy = nullptr;   // copy_state.comp
    std::size_t copy_size = 0;
};

class Engine {
public:
    // Free-energy estimate + normalized peak density from the per-step
    // renormalization (QRhi/GL RelaxStats parity).
    struct RelaxStats {
        double energy = 0.0;
        double peak = 0.0;
    };
    // GPU-reduced norm (sum |psi|^2 dV) and raw peak density.
    struct NormPeak {
        double sum = 0.0;
        double peak = 0.0;
    };

    Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    ~Engine() { destroy(); }

    // half_v / kinetic are SplitOperator3D's phase tables; psi0 the initial
    // field. Cubic grids only (one baked fft_line<n>), like the QRhi engine.
    bool initialize(DeviceContext& ctx, const ses::Grid3D& grid,
                    const EngineKernels& blobs,
                    const std::vector<ses::Complex<double>>& half_v,
                    const std::vector<ses::Complex<double>>& kinetic,
                    const std::vector<ses::Complex<double>>& psi0) {
        ctx_ = &ctx;
        grid_ = grid;
        n_ = grid.x.n;
        cells_ = static_cast<std::size_t>(grid.size());
        cell_volume_ = grid.cell_volume();
        if (grid.y.n != n_ || grid.z.n != n_) {
            std::fprintf(stderr, "ses_vk::Engine: only cubic grids supported\n");
            return false;
        }
        mul_groups_ = static_cast<std::uint32_t>((cells_ + 255) / 256);
        field_bytes_ = 2 * cells_ * sizeof(float);

        // Dynamic-offset UBO slot stride: the device's minimum offset
        // alignment, grown to hold one KickParams block.
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx.phys_dev, &props);
        kick_stride_ = static_cast<std::uint32_t>(
            props.limits.minUniformBufferOffsetAlignment);
        if (kick_stride_ == 0) {
            kick_stride_ = 256;
        }
        while (kick_stride_ < sizeof(KickParams)) {
            kick_stride_ *= 2;
        }

        if (!mul_.create(ctx, blobs.mul, blobs.mul_size,
                         {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                          {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !conj_.create(ctx, blobs.conj, blobs.conj_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !fft_.create(ctx, blobs.fft, blobs.fft_size,
                         {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                          {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !norm_.create(ctx, blobs.norm, blobs.norm_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                           {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !scale_.create(ctx, blobs.scale, blobs.scale_size,
                           {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !kick_.create(ctx, blobs.kick, blobs.kick_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC}}) ||
            !shear_.create(ctx, blobs.shear, blobs.shear_size,
                           {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}}) ||
            !inner_.create(ctx, blobs.inner, blobs.inner_size,
                           {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                            {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !axpy_.create(ctx, blobs.axpy, blobs.axpy_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                           {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}}) ||
            !copy_.create(ctx, blobs.copy, blobs.copy_size,
                          {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
                           {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}})) {
            return false;
        }

        if (!ctx.create_device_buffer(field_bytes_, &psi_) ||
            !ctx.create_device_buffer(field_bytes_, &half_) ||
            !ctx.create_device_buffer(field_bytes_, &kin_) ||
            !ctx.create_device_buffer(2 * kGroups * sizeof(float), &partials_) ||
            !ctx.create_host_buffer(field_bytes_,
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    &staging_)) {
            return false;
        }

        // std140 parameter blocks, written once into host-mapped UBOs (every
        // submission fence-waits, so no in-flight aliasing; the scale UBO is
        // the only one rewritten, between submissions).
        const MulParams muln{static_cast<std::uint32_t>(cells_), 0, 0, 0};
        const ConjParams conj1{static_cast<std::uint32_t>(cells_), 1.0f, 0.0f,
                               0.0f};
        const ConjParams conjN{static_cast<std::uint32_t>(cells_),
                               1.0f / static_cast<float>(cells_), 0.0f, 0.0f};
        const std::int32_t nn = n_ * n_;
        const FftParams fftp[3] = {
            {nn, n_, 0, 1, nn, 0, 0, 0},   // x-lines (contiguous)
            {n_, 1, nn, n_, nn, 0, 0, 0},  // y-lines
            {nn, 1, 0, nn, nn, 0, 0, 0},   // z-lines
        };
        // conjA: the per-axis inverse-FFT scale (1/n, the single transformed
        // axis) used by the shear path.
        const ConjParams conjA{static_cast<std::uint32_t>(cells_),
                               1.0f / static_cast<float>(n_), 0.0f, 0.0f};
        const ShearParams shear_zero{};
        if (!write_ubo(&muln_ubo_, &muln, sizeof(muln)) ||
            !write_ubo(&conj1_ubo_, &conj1, sizeof(conj1)) ||
            !write_ubo(&conjN_ubo_, &conjN, sizeof(conjN)) ||
            !write_ubo(&fft_ubo_[0], &fftp[0], sizeof(FftParams)) ||
            !write_ubo(&fft_ubo_[1], &fftp[1], sizeof(FftParams)) ||
            !write_ubo(&fft_ubo_[2], &fftp[2], sizeof(FftParams)) ||
            !write_ubo(&scale_ubo_, &conj1, sizeof(conj1)) ||
            !write_ubo(&conjA_ubo_, &conjA, sizeof(conjA)) ||
            !write_ubo(&shear_ubo_[0], &shear_zero, sizeof(shear_zero)) ||
            !write_ubo(&shear_ubo_[1], &shear_zero, sizeof(shear_zero))) {
            return false;
        }
        if (!ctx.create_host_buffer(2 * kick_stride_,
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    &kick_ubo_)) {
            return false;
        }
        kick_slots_ = 2;

        // conj (0,0) coefficients: the axpy UBO is rewritten per projection.
        const AxpyParams axpy0{static_cast<std::uint32_t>(cells_), 0, 0.0f,
                               0.0f};
        if (!write_ubo(&axpy_ubo_, &axpy0, sizeof(axpy0))) {
            return false;
        }

        // Descriptor sets: 13 base + 2 relax + 4 per resident state (the
        // harness-scale pool; the atlas stage brings a growable arena).
        if (!arena_.create(ctx_ ? *ctx_ : ctx, 48, 96, 48, 2)) {
            return false;
        }
        half_set_ = make_mul_set(half_.buf);
        kin_set_ = make_mul_set(kin_.buf);
        conj1_set_ = make_unary_set(conj_, conj1_ubo_, sizeof(ConjParams));
        conjN_set_ = make_unary_set(conj_, conjN_ubo_, sizeof(ConjParams));
        for (int a = 0; a < 3; ++a) {
            fft_set_[a] = make_unary_set(fft_, fft_ubo_[a], sizeof(FftParams));
        }
        scale_set_ = make_unary_set(scale_, scale_ubo_, sizeof(ConjParams));
        conjA_set_ = make_unary_set(conj_, conjA_ubo_, sizeof(ConjParams));
        shear_set_[0] = make_unary_set(shear_, shear_ubo_[0], sizeof(ShearParams));
        shear_set_[1] = make_unary_set(shear_, shear_ubo_[1], sizeof(ShearParams));
        kick_set_ = arena_.allocate(*ctx_, kick_.set_layout());
        if (kick_set_ != VK_NULL_HANDLE) {
            arena_.write_buffer(*ctx_, kick_set_, 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, psi_.buf);
            arena_.write_buffer(*ctx_, kick_set_, 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                kick_ubo_.buf, sizeof(KickParams));
        }
        norm_set_ = arena_.allocate(*ctx_, norm_.set_layout());
        if (norm_set_ != VK_NULL_HANDLE) {
            arena_.write_buffer(*ctx_, norm_set_, 0,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, psi_.buf);
            arena_.write_buffer(*ctx_, norm_set_, 1,
                                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, muln_ubo_.buf,
                                sizeof(MulParams));
            arena_.write_buffer(*ctx_, norm_set_, 2,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                partials_.buf);
        }
        if (half_set_ == VK_NULL_HANDLE || kin_set_ == VK_NULL_HANDLE ||
            conj1_set_ == VK_NULL_HANDLE || conjN_set_ == VK_NULL_HANDLE ||
            fft_set_[0] == VK_NULL_HANDLE || fft_set_[1] == VK_NULL_HANDLE ||
            fft_set_[2] == VK_NULL_HANDLE || scale_set_ == VK_NULL_HANDLE ||
            norm_set_ == VK_NULL_HANDLE || conjA_set_ == VK_NULL_HANDLE ||
            shear_set_[0] == VK_NULL_HANDLE || shear_set_[1] == VK_NULL_HANDLE ||
            kick_set_ == VK_NULL_HANDLE) {
            return false;
        }

        return upload_field(half_, half_v) && upload_field(kin_, kinetic) &&
               upload_field(psi_, psi0);
    }

    // psi <- (halfV . IFFT . kin . FFT . halfV)^nsteps psi. One submission;
    // a compute-to-compute barrier precedes every psi-aliasing dispatch.
    void step(int nsteps) {
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        Recorder r{shot.cb(), true};
        for (int s = 0; s < nsteps; ++s) {
            run_step_body(r, half_set_, kin_set_);
        }
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // Driven Strang steps: kick(t) . step . kick(t+dt), theta = amplitude
    // cos(omega t) dt/2. Per-kick thetas differ within the batch, so the kick
    // parameters live in dynamic-offset slots of ONE host-mapped UBO and the
    // whole batch records as a single submission (QRhi single-frame parity).
    void driven_step(const ses::DipoleDrive& d, double t0, double dt,
                     int nsteps) {
        const int kicks = 2 * nsteps;
        if (!ensure_kick_capacity(kicks)) {
            return;
        }
        char* slots = static_cast<char*>(kick_ubo_.mapped);
        for (int s = 0; s < nsteps; ++s) {
            const double t = t0 + s * dt;
            KickParams kp = make_kick_params(
                d.axis, d.amplitude * std::cos(d.omega * t) * 0.5 * dt);
            std::memcpy(slots + static_cast<std::size_t>(2 * s) * kick_stride_,
                        &kp, sizeof(kp));
            kp.theta = static_cast<float>(
                d.amplitude * std::cos(d.omega * (t + dt)) * 0.5 * dt);
            std::memcpy(
                slots + static_cast<std::size_t>(2 * s + 1) * kick_stride_, &kp,
                sizeof(kp));
        }
        vmaFlushAllocation(ctx_->allocator, kick_ubo_.alloc, 0, VK_WHOLE_SIZE);

        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        Recorder r{shot.cb(), true};
        for (int s = 0; s < nsteps; ++s) {
            r.dispatch_dyn(kick_, kick_set_, mul_groups_,
                           static_cast<std::uint32_t>(2 * s) * kick_stride_);
            run_step_body(r, half_set_, kin_set_);
            r.dispatch_dyn(kick_, kick_set_, mul_groups_,
                           static_cast<std::uint32_t>(2 * s + 1) * kick_stride_);
        }
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // Swap the half-potential phase table (e.g. the diamagnetic-augmented one
    // before a magnetic run).
    bool set_half_potential(const std::vector<ses::Complex<double>>& half_v) {
        return upload_field(half_, half_v);
    }

    // Exact three-shear rotation of psi about coordinate `axis` by theta --
    // the GPU transcription of core/rotation.hpp rotate_axis. One submission.
    void rotate_axis_shear(int axis, double theta) {
        const int b = (axis + 1) % 3;  // in-plane axes (b x c = axis)
        const int c = (axis + 2) % 3;
        stage_rotation_ubos(b, c, theta);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        Recorder r{shot.cb(), true};
        record_rotation(r, b, c);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }
    void rotate_z_shear(double theta) { rotate_axis_shear(2, theta); }

    // Magnetic Strang step: R(a) . real-step . R(a), a = (B/2)(dt/2), about
    // the field axis. half_ must hold the diamagnetic-augmented table
    // (set_half_potential). The half-angle is the same for every rotation in
    // the batch, so the two shear parameter sets are staged once and the
    // whole batch records as one submission.
    void magnetic_step(int axis, double half_angle, int nsteps) {
        const int b = (axis + 1) % 3;
        const int c = (axis + 2) % 3;
        stage_rotation_ubos(b, c, half_angle);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        Recorder r{shot.cb(), true};
        for (int s = 0; s < nsteps; ++s) {
            record_rotation(r, b, c);
            run_step_body(r, half_set_, kin_set_);
            record_rotation(r, b, c);
        }
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // Imaginary-time weight tables (packed vec2(w,0)) + dtau/dV for the
    // renormalization; call after initialize().
    bool set_relax_tables(const std::vector<double>& half_w,
                          const std::vector<double>& kin_w, double dtau,
                          double cell_volume) {
        dtau_ = dtau;
        cell_volume_ = cell_volume;
        std::vector<float> hf(2 * cells_, 0.0f);
        std::vector<float> kf(2 * cells_, 0.0f);
        for (std::size_t i = 0; i < cells_; ++i) {
            hf[2 * i] = static_cast<float>(half_w[i]);
            kf[2 * i] = static_cast<float>(kin_w[i]);
        }
        if (!ctx_->create_device_buffer(field_bytes_, &relax_half_) ||
            !ctx_->create_device_buffer(field_bytes_, &relax_kin_)) {
            return false;
        }
        if (!upload_raw(relax_half_, hf.data(), field_bytes_) ||
            !upload_raw(relax_kin_, kf.data(), field_bytes_)) {
            return false;
        }
        relax_half_set_ = make_mul_set(relax_half_.buf);
        relax_kin_set_ = make_mul_set(relax_kin_.buf);
        return relax_half_set_ != VK_NULL_HANDLE &&
               relax_kin_set_ != VK_NULL_HANDLE;
    }

    // e^{-H dtau} Strang steps with per-step renormalization. Each step: one
    // submission for the imaginary body + norm reduction + partials readback,
    // a host-double finish, then a submission scaling by 1/sqrt(norm). The
    // pre-renorm norm decays as e^{-2 E dtau} -> free energy estimate.
    RelaxStats relax_step(int nsteps) {
        RelaxStats stats;
        for (int s = 0; s < nsteps; ++s) {
            OneShot shot;
            if (!shot.begin(*ctx_)) {
                return stats;
            }
            Recorder r{shot.cb(), true};
            run_step_body(r, relax_half_set_, relax_kin_set_);
            barrier_compute_to_compute(shot.cb());
            norm_.bind(shot.cb(), norm_set_);
            vkCmdDispatch(shot.cb(), kGroups, 1, 1);
            barrier_compute_to_transfer(shot.cb());
            const VkBufferCopy down{0, 0, 2 * kGroups * sizeof(float)};
            vkCmdCopyBuffer(shot.cb(), partials_.buf, staging_.buf, 1, &down);
            barrier_transfer_to_host(shot.cb());
            shot.submit_and_wait(*ctx_);
            shot.destroy(*ctx_);
            stats = renormalize_and_estimate();
        }
        return stats;
    }

    // Reset psi from a host field.
    bool upload_state(const std::vector<ses::Complex<double>>& psi) {
        return upload_field(psi_, psi);
    }

    // ---- resident states (one int handle space, QRhi/GL parity) ----------

    // Upload a CPU state into its own resident fp32 buffer; returns a handle
    // usable with every per-state op, or -1 on failure.
    int create_state_buffer(const std::vector<ses::Complex<double>>& state) {
        State st;
        if (!ctx_->create_device_buffer(field_bytes_, &st.buf) ||
            !upload_field(st.buf, state)) {
            ctx_->destroy_buffer(&st.buf);
            return -1;
        }
        st.inner_set = arena_.allocate(*ctx_, inner_.set_layout());
        st.axpy_set = arena_.allocate(*ctx_, axpy_.set_layout());
        st.copy_set = arena_.allocate(*ctx_, copy_.set_layout());
        st.mul_set = arena_.allocate(*ctx_, mul_.set_layout());
        if (st.inner_set == VK_NULL_HANDLE || st.axpy_set == VK_NULL_HANDLE ||
            st.copy_set == VK_NULL_HANDLE || st.mul_set == VK_NULL_HANDLE) {
            ctx_->destroy_buffer(&st.buf);
            return -1;
        }
        const auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        const auto uniform = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        arena_.write_buffer(*ctx_, st.inner_set, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, st.inner_set, 1, uniform, muln_ubo_.buf,
                            sizeof(MulParams));
        arena_.write_buffer(*ctx_, st.inner_set, 2, storage, partials_.buf);
        arena_.write_buffer(*ctx_, st.inner_set, 3, storage, st.buf.buf);
        arena_.write_buffer(*ctx_, st.axpy_set, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, st.axpy_set, 1, uniform, axpy_ubo_.buf,
                            sizeof(AxpyParams));
        arena_.write_buffer(*ctx_, st.axpy_set, 3, storage, st.buf.buf);
        arena_.write_buffer(*ctx_, st.copy_set, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, st.copy_set, 1, uniform, muln_ubo_.buf,
                            sizeof(MulParams));
        arena_.write_buffer(*ctx_, st.copy_set, 3, storage, st.buf.buf);
        arena_.write_buffer(*ctx_, st.mul_set, 0, storage, psi_.buf);
        arena_.write_buffer(*ctx_, st.mul_set, 1, storage, st.buf.buf);
        arena_.write_buffer(*ctx_, st.mul_set, 2, uniform, muln_ubo_.buf,
                            sizeof(MulParams));
        states_.push_back(st);
        return static_cast<int>(states_.size()) - 1;
    }

    // Free a resident state's buffer; the slot stays so handles remain
    // stable (its pool sets are simply abandoned until the arena resets).
    void release_state(int handle) {
        State* st = state_at(handle);
        if (st == nullptr) {
            return;
        }
        vkDeviceWaitIdle(ctx_->device);
        ctx_->destroy_buffer(&st->buf);
    }

    // <state|psi> = sum conj(state)*psi * dV.
    ses::Complex<double> inner_with_psi(int handle) {
        State* st = state_at(handle);
        if (st == nullptr) {
            return {};
        }
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return {};
        }
        inner_.bind(shot.cb(), st->inner_set);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_partials_readback(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return {};
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        double re = 0.0;
        double im = 0.0;
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            re += p[2 * g];
            im += p[2 * g + 1];
        }
        return ses::Complex<double>{re * cell_volume_, im * cell_volume_};
    }

    // Deflated imaginary-time relax: imaginary Strang body, Gram-Schmidt
    // project-out of every `lower` state (psi -= <phi|psi> phi), renorm.
    RelaxStats relax_deflated_step(const std::vector<int>& lower, int nsteps) {
        RelaxStats stats;
        for (int s = 0; s < nsteps; ++s) {
            OneShot shot;
            if (!shot.begin(*ctx_)) {
                return stats;
            }
            Recorder r{shot.cb(), true};
            run_step_body(r, relax_half_set_, relax_kin_set_);
            shot.submit_and_wait(*ctx_);
            shot.destroy(*ctx_);
            for (int h : lower) {
                const ses::Complex<double> c = inner_with_psi(h);
                subtract_projection(h, c.real(), c.imag());
            }
            stats = renormalize_and_estimate();
        }
        return stats;
    }

    // psi <- src (bitwise; the quantum-jump collapse path).
    void copy_into_psi(int handle) {
        State* st = state_at(handle);
        if (st == nullptr) {
            return;
        }
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        copy_.bind(shot.cb(), st->copy_set);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // psi <- psi * state (elementwise): the absorbing-boundary damp.
    void apply_mask(int handle) {
        State* st = state_at(handle);
        if (st == nullptr) {
            return;
        }
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        mul_.bind(shot.cb(), st->mul_set);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // GPU-reduced norm (sum |psi|^2 dV) and raw peak density.
    NormPeak norm_and_peak() {
        NormPeak out;
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return out;
        }
        norm_.bind(shot.cb(), norm_set_);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_partials_readback(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return out;
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            out.sum += p[2 * g];
            out.peak = std::max(out.peak, static_cast<double>(p[2 * g + 1]));
        }
        out.sum *= cell_volume_;
        return out;
    }

    // psi <- s * psi (fp32 drift renormalization).
    void scale(float s) {
        const ConjParams sp{static_cast<std::uint32_t>(cells_), s, 0.0f, 0.0f};
        std::memcpy(scale_ubo_.mapped, &sp, sizeof(sp));
        vmaFlushAllocation(ctx_->allocator, scale_ubo_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        scale_.bind(shot.cb(), scale_set_);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // Interleaved RG floats, 2 per cell.
    bool readback(std::vector<float>& out) {
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return false;
        }
        barrier_compute_to_transfer(shot.cb());
        const VkBufferCopy down{0, 0, field_bytes_};
        vkCmdCopyBuffer(shot.cb(), psi_.buf, staging_.buf, 1, &down);
        barrier_transfer_to_host(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return false;
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        out.assign(p, p + 2 * cells_);
        return true;
    }

    void destroy() {
        if (ctx_ == nullptr) {
            return;
        }
        if (ctx_->device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(ctx_->device);
        }
        for (State& st : states_) {
            ctx_->destroy_buffer(&st.buf);
        }
        states_.clear();
        arena_.destroy(*ctx_);
        copy_.destroy(*ctx_);
        axpy_.destroy(*ctx_);
        inner_.destroy(*ctx_);
        shear_.destroy(*ctx_);
        kick_.destroy(*ctx_);
        scale_.destroy(*ctx_);
        norm_.destroy(*ctx_);
        fft_.destroy(*ctx_);
        conj_.destroy(*ctx_);
        mul_.destroy(*ctx_);
        ctx_->destroy_buffer(&relax_kin_);
        ctx_->destroy_buffer(&relax_half_);
        ctx_->destroy_buffer(&axpy_ubo_);
        ctx_->destroy_buffer(&kick_ubo_);
        ctx_->destroy_buffer(&shear_ubo_[1]);
        ctx_->destroy_buffer(&shear_ubo_[0]);
        ctx_->destroy_buffer(&conjA_ubo_);
        ctx_->destroy_buffer(&scale_ubo_);
        for (int a = 0; a < 3; ++a) {
            ctx_->destroy_buffer(&fft_ubo_[a]);
        }
        ctx_->destroy_buffer(&conjN_ubo_);
        ctx_->destroy_buffer(&conj1_ubo_);
        ctx_->destroy_buffer(&muln_ubo_);
        ctx_->destroy_buffer(&staging_);
        ctx_->destroy_buffer(&partials_);
        ctx_->destroy_buffer(&kin_);
        ctx_->destroy_buffer(&half_);
        ctx_->destroy_buffer(&psi_);
        ctx_ = nullptr;
    }

private:
    static constexpr std::uint32_t kGroups = 256;

    struct alignas(16) MulParams {
        std::uint32_t n, p0, p1, p2;
    };
    struct alignas(16) ConjParams {
        std::uint32_t n;
        float scale;
        float p0, p1;
    };
    struct alignas(16) FftParams {
        std::int32_t mod_a, mul_b, mul_c, stride, n_lines, p0, p1, p2;
    };
    // std140 {uint n; vec2 c}: c aligns to offset 8 (uint + 4 bytes pad).
    struct alignas(16) AxpyParams {
        std::uint32_t n, pad;
        float cx, cy;
    };
    struct alignas(16) KickParams {
        std::uint32_t n;
        std::int32_t nx, ny;
        float theta;
        float box_min[4];
        float cell_h[4];
        float axis[4];
    };
    // std140 all-scalar block (tight 4-byte packing), matches shear.comp;
    // the trailing pad makes the 16-byte alignment explicit (dodges C4324).
    struct alignas(16) ShearParams {
        std::uint32_t n;
        std::int32_t nx, ny, nz;
        std::int32_t freq_axis, coord_axis, nf;
        float kscale, cmin, ch, coeff;
        float pad0;
    };

    // Emits the compute-to-compute hazard edge before every dispatch except
    // the first of a command buffer (prior submissions are fence-complete).
    struct Recorder {
        VkCommandBuffer cb;
        bool first;
        void dispatch(const Kernel& k, VkDescriptorSet set,
                      std::uint32_t groups) {
            if (!first) {
                barrier_compute_to_compute(cb);
            }
            first = false;
            k.bind(cb, set);
            vkCmdDispatch(cb, groups, 1, 1);
        }
        void dispatch_dyn(const Kernel& k, VkDescriptorSet set,
                          std::uint32_t groups, std::uint32_t offset) {
            if (!first) {
                barrier_compute_to_compute(cb);
            }
            first = false;
            k.bind(cb, set, offset);
            vkCmdDispatch(cb, groups, 1, 1);
        }
    };

    static std::vector<float> to_rg32f(
        const std::vector<ses::Complex<double>>& src) {
        std::vector<float> out(2 * src.size());
        for (std::size_t i = 0; i < src.size(); ++i) {
            out[2 * i] = static_cast<float>(src[i].real());
            out[2 * i + 1] = static_cast<float>(src[i].imag());
        }
        return out;
    }

    bool write_ubo(Buffer* ubo, const void* data, std::size_t size) {
        if (!ctx_->create_host_buffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      ubo)) {
            return false;
        }
        std::memcpy(ubo->mapped, data, size);
        vmaFlushAllocation(ctx_->allocator, ubo->alloc, 0, VK_WHOLE_SIZE);
        return true;
    }

    VkDescriptorSet make_mul_set(VkBuffer table) {
        VkDescriptorSet set = arena_.allocate(*ctx_, mul_.set_layout());
        if (set == VK_NULL_HANDLE) {
            return set;
        }
        arena_.write_buffer(*ctx_, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            psi_.buf);
        arena_.write_buffer(*ctx_, set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            table);
        arena_.write_buffer(*ctx_, set, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            muln_ubo_.buf, sizeof(MulParams));
        return set;
    }

    VkDescriptorSet make_unary_set(const Kernel& k, const Buffer& ubo,
                                   std::size_t ubo_size) {
        VkDescriptorSet set = arena_.allocate(*ctx_, k.set_layout());
        if (set == VK_NULL_HANDLE) {
            return set;
        }
        arena_.write_buffer(*ctx_, set, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            psi_.buf);
        arena_.write_buffer(*ctx_, set, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            ubo.buf, ubo_size);
        return set;
    }

    bool upload_field(Buffer& dst,
                      const std::vector<ses::Complex<double>>& src) {
        const std::vector<float> f = to_rg32f(src);
        return upload_raw(dst, f.data(), f.size() * sizeof(float));
    }

    bool upload_raw(Buffer& dst, const void* data, VkDeviceSize bytes) {
        std::memcpy(staging_.mapped, data, bytes);
        vmaFlushAllocation(ctx_->allocator, staging_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return false;
        }
        const VkBufferCopy up{0, 0, bytes};
        vkCmdCopyBuffer(shot.cb(), staging_.buf, dst.buf, 1, &up);
        barrier_transfer_to_compute(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        return ok;
    }

    // A resident state: its fp32 buffer + the four per-op descriptor sets
    // (inner/axpy/copy/mask-multiply against live psi).
    struct State {
        Buffer buf{};
        VkDescriptorSet inner_set = VK_NULL_HANDLE;
        VkDescriptorSet axpy_set = VK_NULL_HANDLE;
        VkDescriptorSet copy_set = VK_NULL_HANDLE;
        VkDescriptorSet mul_set = VK_NULL_HANDLE;
    };

    State* state_at(int handle) {
        if (handle < 0 || handle >= static_cast<int>(states_.size())) {
            return nullptr;
        }
        State* st = &states_[static_cast<std::size_t>(handle)];
        return (st->buf.buf == VK_NULL_HANDLE) ? nullptr : st;
    }

    // Record: compute -> transfer edge, partials -> staging copy, host edge.
    void record_partials_readback(VkCommandBuffer cb) {
        barrier_compute_to_transfer(cb);
        const VkBufferCopy down{0, 0, 2 * kGroups * sizeof(float)};
        vkCmdCopyBuffer(cb, partials_.buf, staging_.buf, 1, &down);
        barrier_transfer_to_host(cb);
    }

    // psi -= (cre + i cim) * state (the Gram-Schmidt projection subtract).
    void subtract_projection(int handle, double cre, double cim) {
        State* st = state_at(handle);
        if (st == nullptr) {
            return;
        }
        const AxpyParams ap{static_cast<std::uint32_t>(cells_), 0,
                            static_cast<float>(cre), static_cast<float>(cim)};
        std::memcpy(axpy_ubo_.mapped, &ap, sizeof(ap));
        vmaFlushAllocation(ctx_->allocator, axpy_ubo_.alloc, 0, VK_WHOLE_SIZE);
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return;
        }
        axpy_.bind(shot.cb(), st->axpy_set);
        vkCmdDispatch(shot.cb(), mul_groups_, 1, 1);
        shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
    }

    // Norm reduction + readback -> host finish -> 1/sqrt(norm) scale. The
    // pre-renorm norm decays as e^{-2 E dtau} -> free-energy estimate.
    RelaxStats renormalize_and_estimate() {
        RelaxStats stats;
        OneShot shot;
        if (!shot.begin(*ctx_)) {
            return stats;
        }
        norm_.bind(shot.cb(), norm_set_);
        vkCmdDispatch(shot.cb(), kGroups, 1, 1);
        record_partials_readback(shot.cb());
        const bool ok = shot.submit_and_wait(*ctx_);
        shot.destroy(*ctx_);
        if (!ok) {
            return stats;
        }
        vmaInvalidateAllocation(ctx_->allocator, staging_.alloc, 0,
                                VK_WHOLE_SIZE);
        const float* p = static_cast<const float*>(staging_.mapped);
        double sum = 0.0;
        double peak = 0.0;
        for (std::uint32_t g = 0; g < kGroups; ++g) {
            sum += p[2 * g];
            peak = std::max(peak, static_cast<double>(p[2 * g + 1]));
        }
        const double norm_sq = sum * cell_volume_;
        const double inv = (norm_sq > 0.0) ? 1.0 / std::sqrt(norm_sq) : 0.0;
        stats.energy = (norm_sq > 0.0 && dtau_ > 0.0)
                           ? -std::log(norm_sq) / (2.0 * dtau_)
                           : 0.0;
        stats.peak = (norm_sq > 0.0) ? peak / norm_sq : 0.0;
        scale(static_cast<float>(inv));
        return stats;
    }

    KickParams make_kick_params(const ses::Vec3d& axis, double theta) const {
        KickParams kp{};
        kp.n = static_cast<std::uint32_t>(cells_);
        kp.nx = grid_.x.n;
        kp.ny = grid_.y.n;
        kp.theta = static_cast<float>(theta);
        kp.box_min[0] = static_cast<float>(grid_.x.xmin);
        kp.box_min[1] = static_cast<float>(grid_.y.xmin);
        kp.box_min[2] = static_cast<float>(grid_.z.xmin);
        kp.cell_h[0] = static_cast<float>(grid_.x.spacing());
        kp.cell_h[1] = static_cast<float>(grid_.y.spacing());
        kp.cell_h[2] = static_cast<float>(grid_.z.spacing());
        kp.axis[0] = static_cast<float>(axis.x);
        kp.axis[1] = static_cast<float>(axis.y);
        kp.axis[2] = static_cast<float>(axis.z);
        return kp;
    }

    // Grow the dynamic-offset kick UBO to `kicks` slots and re-point the
    // descriptor set at the new buffer (rewriting a fence-idle set is legal).
    bool ensure_kick_capacity(int kicks) {
        const VkDeviceSize need =
            static_cast<VkDeviceSize>(kick_stride_) * kicks;
        if (kick_ubo_.buf != VK_NULL_HANDLE && kick_slots_ >= kicks) {
            return true;
        }
        ctx_->destroy_buffer(&kick_ubo_);
        if (!ctx_->create_host_buffer(
                need, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &kick_ubo_)) {
            return false;
        }
        kick_slots_ = kicks;
        arena_.write_buffer(*ctx_, kick_set_, 1,
                            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                            kick_ubo_.buf, sizeof(KickParams));
        return true;
    }

    const ses::Grid1D& axis_grid(int a) const {
        return a == 0 ? grid_.x : (a == 1 ? grid_.y : grid_.z);
    }

    // Shearing lines along freq_axis (frequency space along freq_axis),
    // shift each line by coeff * (its coord_axis coordinate).
    ShearParams make_shear_params(int freq_axis, int coord_axis,
                                  double coeff) const {
        const ses::Grid1D& fa = axis_grid(freq_axis);
        const ses::Grid1D& ca = axis_grid(coord_axis);
        const double two_pi = 6.283185307179586;
        ShearParams sp{};
        sp.n = static_cast<std::uint32_t>(cells_);
        sp.nx = grid_.x.n;
        sp.ny = grid_.y.n;
        sp.nz = grid_.z.n;
        sp.freq_axis = freq_axis;
        sp.coord_axis = coord_axis;
        sp.nf = fa.n;
        sp.kscale = static_cast<float>(two_pi / (fa.xmax - fa.xmin));
        sp.cmin = static_cast<float>(ca.xmin);
        sp.ch = static_cast<float>(ca.spacing());
        sp.coeff = static_cast<float>(coeff);
        return sp;
    }

    // Write the two shear parameter sets one (or many) three-shear
    // rotation(s) about the axis perpendicular to (b, c) need: set0 =
    // (b, c, -tan(theta/2)) used twice, set1 = (c, b, sin(theta)).
    void stage_rotation_ubos(int b, int c, double theta) {
        const double t = std::tan(0.5 * theta);
        const double sn = std::sin(theta);
        const ShearParams s0 = make_shear_params(b, c, -t);
        const ShearParams s1 = make_shear_params(c, b, sn);
        std::memcpy(shear_ubo_[0].mapped, &s0, sizeof(s0));
        std::memcpy(shear_ubo_[1].mapped, &s1, sizeof(s1));
        vmaFlushAllocation(ctx_->allocator, shear_ubo_[0].alloc, 0,
                           VK_WHOLE_SIZE);
        vmaFlushAllocation(ctx_->allocator, shear_ubo_[1].alloc, 0,
                           VK_WHOLE_SIZE);
    }

    // One staged shear: FFT along freq_axis, phase-shift (shear_set_[which]),
    // then the inverse FFT along that axis (conj -> FFT -> conj/n).
    void record_shear(Recorder& r, int which, int freq_axis) {
        const std::uint32_t nn = static_cast<std::uint32_t>(n_) *
                                 static_cast<std::uint32_t>(n_);
        r.dispatch(fft_, fft_set_[freq_axis], nn);
        r.dispatch(shear_, shear_set_[which], mul_groups_);
        r.dispatch(conj_, conj1_set_, mul_groups_);
        r.dispatch(fft_, fft_set_[freq_axis], nn);
        r.dispatch(conj_, conjA_set_, mul_groups_);
    }

    // One full three-shear rotation (set0 on b, set1 on c, set0 on b).
    void record_rotation(Recorder& r, int b, int c) {
        record_shear(r, 0, b);
        record_shear(r, 1, c);
        record_shear(r, 0, b);
    }

    // halfV . IFFT . kin . FFT . halfV (the inverse FFT = conj . FFT .
    // conj/N) -- the QRhi engine's hand-rolled chain, barriers explicit.
    void run_step_body(Recorder& r, VkDescriptorSet half_set,
                       VkDescriptorSet kin_set) {
        r.dispatch(mul_, half_set, mul_groups_);
        fft3(r);
        r.dispatch(mul_, kin_set, mul_groups_);
        r.dispatch(conj_, conj1_set_, mul_groups_);
        fft3(r);
        r.dispatch(conj_, conjN_set_, mul_groups_);
        r.dispatch(mul_, half_set, mul_groups_);
    }

    void fft3(Recorder& r) {
        const std::uint32_t nn = static_cast<std::uint32_t>(n_) *
                                 static_cast<std::uint32_t>(n_);
        for (int a = 0; a < 3; ++a) {
            r.dispatch(fft_, fft_set_[a], nn);
        }
    }

    DeviceContext* ctx_ = nullptr;
    ses::Grid3D grid_{};
    int n_ = 0;
    std::size_t cells_ = 0;
    std::uint32_t mul_groups_ = 0;
    VkDeviceSize field_bytes_ = 0;
    double cell_volume_ = 1.0;
    double dtau_ = 0.0;
    std::uint32_t kick_stride_ = 256;
    int kick_slots_ = 0;

    Kernel mul_;
    Kernel conj_;
    Kernel fft_;
    Kernel norm_;
    Kernel scale_;
    Kernel kick_;
    Kernel shear_;
    Kernel inner_;
    Kernel axpy_;
    Kernel copy_;
    DescriptorArena arena_;
    std::vector<State> states_;

    Buffer psi_{};
    Buffer half_{};
    Buffer kin_{};
    Buffer partials_{};
    Buffer staging_{};
    Buffer muln_ubo_{};
    Buffer conj1_ubo_{};
    Buffer conjN_ubo_{};
    Buffer fft_ubo_[3]{};
    Buffer scale_ubo_{};
    Buffer conjA_ubo_{};
    Buffer shear_ubo_[2]{};
    Buffer kick_ubo_{};
    Buffer axpy_ubo_{};
    Buffer relax_half_{};
    Buffer relax_kin_{};

    VkDescriptorSet half_set_ = VK_NULL_HANDLE;
    VkDescriptorSet kin_set_ = VK_NULL_HANDLE;
    VkDescriptorSet conj1_set_ = VK_NULL_HANDLE;
    VkDescriptorSet conjN_set_ = VK_NULL_HANDLE;
    VkDescriptorSet fft_set_[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE,
                                   VK_NULL_HANDLE};
    VkDescriptorSet norm_set_ = VK_NULL_HANDLE;
    VkDescriptorSet scale_set_ = VK_NULL_HANDLE;
    VkDescriptorSet conjA_set_ = VK_NULL_HANDLE;
    VkDescriptorSet shear_set_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet kick_set_ = VK_NULL_HANDLE;
    VkDescriptorSet relax_half_set_ = VK_NULL_HANDLE;
    VkDescriptorSet relax_kin_set_ = VK_NULL_HANDLE;
};

}  // namespace ses_vk
