// psi = (u(r)/r) * Y_lm, real harmonics l=0..5
// std140 UBO: box_min/cell_h are vec3 in .xyz (vec4-padded)
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
    int l;
    float4 box_min;
    float4 cell_h;
    int m;
    int n_radial;
    float h_radial;  // = rmax/(n_radial+1)
    float rmax;
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
    float3 r3 = cell_pos(ijk, params.box_min.xyz, params.cell_h.xyz);
    float r = length(r3);
    RadialBin rb = radial_bin(r, params.h_radial);
    float u_over_r = 0.0;
    if (rb.origin || r < params.rmax) {
        u_over_r = radial_u_over_r(u, 0, rb, r, params.h_radial, params.n_radial);
    }
    psi[idx] = Complex(u_over_r * real_Ylm(params.l, params.m, r3.x, r3.y, r3.z), 0.0);
}
