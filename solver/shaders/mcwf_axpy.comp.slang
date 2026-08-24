// Fused MCWF no-jump update: psi[i] += sum_s coef_s * (u_s(|r|)/|r|) * Y_{l_s m_s} * inv_norm_s.
import complex;
import ylm;
import grid;
import radial_sample;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(5, 0)]] StructuredBuffer<float, Std430DataLayout> u;

struct Params {
    uint n;
    int nx;
    int ny;
    int n_states;
    float4 box_min;  // .xyz
    float4 cell_h;   // .xyz
    int n_radial;    // slot stride in floats
    float h_radial;
    float rmax;
    float pad0;
    int4 lm[8];      // .x = l, .y = m
    float4 coef[8];  // .x = cre, .y = cim, .z = inv_norm
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= params.n) {
        return;
    }
    int3 ijk = unflatten_moddiv(idx, params.nx, params.ny);
    float3 p = cell_pos(ijk, params.box_min.xyz, params.cell_h.xyz);
    float r = length(p);
    RadialBin rb = radial_bin(r, params.h_radial);
    Complex add = Complex(0.0, 0.0);
    if (rb.origin || r < params.rmax) {
        for (int s = 0; s < params.n_states; ++s) {
            int base = s * params.n_radial;
            float uor =
                radial_u_over_r(u, base, rb, r, params.h_radial, params.n_radial);
            float R = uor * real_Ylm(params.lm[s].x, params.lm[s].y, p.x, p.y, p.z) *
                      params.coef[s].z;
            add = add + Complex(params.coef[s].xy) * R;
        }
    }
    psi[idx] = psi[idx] + add;
}
