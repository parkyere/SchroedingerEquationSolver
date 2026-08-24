// MC tri-count pass: one workgroup per 8^3 cell block, one thread per cell.
// Block total via shared int atomic (order-independent sum) -> block_sums[wg]
// for the downstream scan pass.
// mc_tab from ses.mc.tables core module; layout named in mc_common.
import grid;
import mc_common;

[[vk::binding(0, 0)]] StructuredBuffer<float, Std430DataLayout> den;
[[vk::binding(1, 0)]] StructuredBuffer<int, Std430DataLayout> mc_tab;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> block_sums;

struct Params {
    int4 npts;
    int4 nblk;
    float4 box_min;
    float4 cell_h;
    float iso;
    int max_tris;
};
[[vk::binding(3, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

groupshared uint s_total;

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 grp : SV_GroupID, uint lidx : SV_GroupIndex) {
    if (lidx == 0u) {
        s_total = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    int wg = int(grp.x);
    // block index via moddiv (nblk power-agnostic), then cell within the 8^3 block.
    int3 blk = unflatten_moddiv(grp.x, params.nblk.x, params.nblk.y);
    int3 cell = blk * 8 + unflatten_moddiv(lidx, 8, 8);

    if (all(cell < (params.npts.xyz - 1))) {
        int cube = mc_classify_cube(den, cell, params.npts.xyz, params.iso);
        uint tris = uint(mc_tab[kMcCountOff + cube]);
        if (tris > 0u) {
            InterlockedAdd(s_total, tris);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lidx == 0u) {
        block_sums[wg] = s_total;
    }
}
