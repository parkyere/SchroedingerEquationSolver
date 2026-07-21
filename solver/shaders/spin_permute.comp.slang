// Qubit relabel of the exact 2^16 state: dst[j] = src[i], where index i sets
// bit perm(b) exactly when j sets bit b. One thread per amplitude; writes are
// coalesced (sequential j), the scattered read is the transpose cost. perm is
// packed 4 bits/qubit into two uints (b<8 -> plo, else phi). This is the
// local/global reorder pass. CONTRACT: vkcheck check_spin_permute.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> src;
[[vk::binding(1, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> dst;

struct Params {
    uint n;
    uint plo;
    uint phi;
    uint pad;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint j = tid.x;
    if (j >= params.n) {
        return;
    }
    uint i = 0u;
    for (uint b = 0u; b < 16u; ++b) {
        if (((j >> b) & 1u) != 0u) {
            uint pb = (b < 8u) ? ((params.plo >> (4u * b)) & 0xFu)
                               : ((params.phi >> (4u * (b - 8u))) & 0xFu);
            i |= 1u << pb;
        }
    }
    dst[j] = src[i];
}
