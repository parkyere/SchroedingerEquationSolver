// Per-nucleus Ehrenfest forces F_k = sum |psi|^2 grad V_k with the ANALYTIC
// gradient of the band-limited Coulomb (ses::coulomb_mean_force verbatim):
// the exact -dE/dc of the lattice energy. Per-workgroup partials {F1, F2} as
// two float4 (w unused).
#include "reduce.slang"
import complex;
import two_center;

[[vk::binding(0, 0)]] StructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(1, 0)]] ConstantBuffer<TwoCenterParams, Std140DataLayout> params;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4, Std430DataLayout> partials;

groupshared float4 s1[256];
groupshared float4 s2[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID,
          uint tid : SV_GroupIndex,
          uint3 grp : SV_GroupID) {
    uint stride = WorkgroupCount().x * 256u;
    float4 a1 = float4(0.0, 0.0, 0.0, 0.0);
    float4 a2 = float4(0.0, 0.0, 0.0, 0.0);
    uint nx = params.nx;
    uint ny = params.ny;
    float z = params.c1.w;
    float h = params.cell_h.w;
    for (uint i = gid.x; i < params.n; i += stride) {
        float w = norm2(psi[i]);
        uint ix = i % nx;
        uint iy = (i / nx) % ny;
        uint iz = i / (nx * ny);
        float3 r = tc_point(params, ix, iy, iz);
        a1 += w * float4(tc_force_density(r, params.c1.xyz, z, h), 0.0);
        a2 += w * float4(tc_force_density(r, params.c2.xyz, z, h), 0.0);
    }
    s1[tid] = a1;
    s2[tid] = a2;
#define TC_FOLD(a, b) { s1[a] += s1[b]; s2[a] += s2[b]; }
    REDUCE_FOLD_256(TC_FOLD, tid);
    if (tid == 0u) {
        partials[2u * grp.x] = s1[0];
        partials[2u * grp.x + 1u] = s2[0];
    }
}
