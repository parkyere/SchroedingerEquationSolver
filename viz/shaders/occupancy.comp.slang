// Empty-space skipping: one workgroup per occ cell, one thread per fine texel.
// Max is associative in fp, so the parallel reduction order is exact.
import viz_ubo;

[[vk::binding(0, 0)]] [[vk::image_format("r32f")]] RWTexture3D<float4> occ;

[[vk::binding(1, 0)]] ConstantBuffer<Ubo, Std140DataLayout> ubo;

[[vk::binding(2, 0)]] Sampler3D<float4> psi_tex;

static const uint kWg = 8;
static const uint kWgVol = kWg * kWg * kWg;

groupshared float s_max[kWgVol];

[shader("compute")]
[numthreads(kWg, kWg, kWg)]
void main(uint3 gwg : SV_GroupID, uint3 glid : SV_GroupThreadID,
          uint lidx : SV_GroupIndex) {
    // One workgroup per occ cell: no bounds check; all invocations must reach the barriers.
    int3 c = int3(gwg);
    uint ow, oh, od;
    occ.GetDimensions(ow, oh, od);
    int3 osize = int3(ow, oh, od);
    uint fw, fh, fd;
    psi_tex.GetDimensions(fw, fh, fd);
    int3 fsize = int3(fw, fh, fd);
    // Proportional range, not fsize/osize: a thin axis would empty the block -> occ==0 -> cloud skipped.
    int3 begin = (c * fsize) / osize;
    int3 end = max(begin + int3(1, 1, 1), ((c + int3(1, 1, 1)) * fsize) /
                                              osize);
    // Grid-stride: the block can exceed the kWg^3 workgroup.
    float m = 0.0;
    for (int k = begin.z + int(glid.z); k < end.z; k += int(kWg)) {
        for (int j = begin.y + int(glid.y); j < end.y; j += int(kWg)) {
            for (int i = begin.x + int(glid.x); i < end.x; i += int(kWg)) {
                float2 s = psi_tex.Load(int4(int3(i, j, k), 0)).xy;
                m = max(m, dot(s, s));
            }
        }
    }
    uint tid = lidx;
    s_max[tid] = m;
    GroupMemoryBarrierWithGroupSync();
    for (uint h = kWgVol / 2u; h > 0u; h >>= 1u) {
        if (tid < h) {
            s_max[tid] = max(s_max[tid], s_max[tid + h]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        occ[c] = float4(s_max[0] * ubo.inv_peak, 0.0, 0.0, 0.0);
    }
}
