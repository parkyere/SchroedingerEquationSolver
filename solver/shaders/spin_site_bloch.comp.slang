// Per-site reduced Bloch vector <sigma_i> over the exact 2^N state: one
// workgroup per site (SV_GroupID.x), grid-stride over half_n bit-clear/bit-set
// amplitude pairs, groupshared tree sum. Mirrors ses::exact_site_bloch bit for
// bit (mul_conj term order). out[3*site+0..2] = (<sx>, <sy>, <sz>).
// CONTRACT: vkcheck check_spin_bloch.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> state;
[[vk::binding(1, 0)]] RWStructuredBuffer<float, Std430DataLayout> outv;

struct Params {
    uint half_n;
    uint pad0;
    uint pad1;
    uint pad2;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

groupshared float red_x[256];
groupshared float red_y[256];
groupshared float red_z[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint site = gid.x;
    uint bit = 1u << site;
    uint local = gtid.x;
    float ax = 0.0;
    float ay = 0.0;
    float az = 0.0;
    for (uint p = local; p < params.half_n; p += 256u) {
        // Insert 0 bit at `site` -> the bit-clear pair index lo (partner hi).
        uint lo = ((p & ~(bit - 1u)) << 1u) | (p & (bit - 1u));
        uint hi = lo | bit;
        Complex clo = state[lo];
        Complex chi = state[hi];
        Complex cr = mul_conj(clo, chi);  // conj(clo) * chi
        ax += cr.re;
        ay += cr.im;
        az += norm2(clo) - norm2(chi);
    }
    red_x[local] = ax;
    red_y[local] = ay;
    red_z[local] = az;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (local < s) {
            red_x[local] += red_x[local + s];
            red_y[local] += red_y[local + s];
            red_z[local] += red_z[local + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (local == 0u) {
        outv[3u * site + 0u] = 2.0 * red_x[0];
        outv[3u * site + 1u] = 2.0 * red_y[0];
        outv[3u * site + 2u] = red_z[0];
    }
}
