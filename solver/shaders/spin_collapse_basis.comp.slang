// Collapse the exact state to the single basis vector |m> (m from spin_born_sample).
// CONTRACT: vkcheck check_spin_measure_exact.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> state;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint, Std430DataLayout> in_m;

struct Params {
    uint n;
    uint pad0, pad1, pad2;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint j = tid.x;
    if (j >= params.n) {
        return;
    }
    state[j] = (j == in_m[0]) ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
}
