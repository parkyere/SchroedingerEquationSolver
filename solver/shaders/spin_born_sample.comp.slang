// Born-sample one basis state from |c_m|^2 over the exact 2^16 register (a full
// projective z-measurement collapses to a single basis state). One workgroup of
// 256 threads, each owning a contiguous chunk of 256 amplitudes: chunk sums ->
// serial prefix -> the chunk holding threshold u*total scans for m. out_m = m.
// CONTRACT: vkcheck check_spin_measure_exact.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> state;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint, Std430DataLayout> out_m;

struct Params {
    float u;
    uint pad0, pad1, pad2;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

groupshared float parts[256];
groupshared float offs[256];
groupshared float gtotal;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID) {
    uint t = gtid.x;
    uint base = t * 256u;
    float p = 0.0;
    for (uint k = 0u; k < 256u; ++k) {
        p += norm2(state[base + k]);
    }
    parts[t] = p;
    GroupMemoryBarrierWithGroupSync();
    if (t == 0u) {
        out_m[0] = 65535u;  // default: u rounds to the top of the distribution
        float acc = 0.0;
        for (uint i = 0u; i < 256u; ++i) {
            offs[i] = acc;
            acc += parts[i];
        }
        gtotal = acc;
    }
    GroupMemoryBarrierWithGroupSync();
    float thr = params.u * gtotal;
    if (offs[t] <= thr && thr < offs[t] + parts[t]) {
        float acc = offs[t];
        uint m = base + 255u;
        for (uint k = 0u; k < 256u; ++k) {
            acc += norm2(state[base + k]);
            if (acc > thr) {
                m = base + k;
                break;
            }
        }
        out_m[0] = m;
    }
}
