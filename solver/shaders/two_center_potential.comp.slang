// Rebuild the R32 potential IN PLACE as V_1 + V_2 (two_center module);
// half_mul reads V live, so the next step already sees the new axis.
import two_center;

[[vk::binding(0, 0)]] RWStructuredBuffer<float, Std430DataLayout> v;
[[vk::binding(1, 0)]] ConstantBuffer<TwoCenterParams, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.n) {
        return;
    }
    uint ix = i % params.nx;
    uint iy = (i / params.nx) % params.ny;
    uint iz = i / (params.nx * params.ny);
    float2 vv = tc_pair(params, tc_point(params, ix, iy, iz));
    v[i] = vv.x + vv.y;
}
