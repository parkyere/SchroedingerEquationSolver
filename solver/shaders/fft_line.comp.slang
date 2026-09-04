// Radix-2 in-place line FFT, unnormalized; one workgroup per line, N/2 threads
// x 2-elem butterfly in groupshared. SPIR-V baked offline per size via -DFFT_N
// (8/64/256/512). Params enumerate lines along an axis; n_lines=1,stride=1,
// base=0 = plain 1-D FFT.
#ifndef FFT_N
#define FFT_N 256
#endif
#if FFT_N == 8
#define FFT_LOG2N 3
#define FFT_NHALF 4
#elif FFT_N == 64
#define FFT_LOG2N 6
#define FFT_NHALF 32
#elif FFT_N == 128
#define FFT_LOG2N 7
#define FFT_NHALF 64
#elif FFT_N == 256
#define FFT_LOG2N 8
#define FFT_NHALF 128
#elif FFT_N == 512
#define FFT_LOG2N 9
#define FFT_NHALF 256
#else
#error "FFT_N must be 8/64/128/256/512"
#endif

import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> data;

struct Params {
    int mod_a;
    int mul_b;
    int mul_c;
    int stride;
    int n_lines;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

groupshared Complex line_sm[FFT_N];

uint bit_reverse(uint v) { return reversebits(v) >> uint(32 - FFT_LOG2N); }

[shader("compute")]
[numthreads(FFT_NHALF, 1, 1)]
void main(uint3 grp : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    int l = int(grp.x);
    if (l >= params.n_lines) {
        return;
    }
    int base = (l % params.mod_a) * params.mul_b + (l / params.mod_a) * params.mul_c;
    uint t = lid.x;
    uint i0 = t;
    uint i1 = t + uint(FFT_NHALF);

    line_sm[i0] = data[base + int(bit_reverse(i0)) * params.stride];
    line_sm[i1] = data[base + int(bit_reverse(i1)) * params.stride];
    GroupMemoryBarrierWithGroupSync();

    for (uint len = 2u; len <= uint(FFT_N); len <<= 1u) {
        uint half_len = len >> 1u;
        uint j = t % half_len;
        uint pos = (t / half_len) * len + j;
        Complex w = cis(-kTwoPi * float(j) / float(len));
        Complex u = line_sm[pos];
        Complex v = line_sm[pos + half_len] * w;
        line_sm[pos] = u + v;
        line_sm[pos + half_len] = u - v;
        GroupMemoryBarrierWithGroupSync();
    }

    data[base + int(i0) * params.stride] = line_sm[i0];
    data[base + int(i1) * params.stride] = line_sm[i1];
}
