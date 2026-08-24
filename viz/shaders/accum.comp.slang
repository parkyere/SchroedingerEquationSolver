// Sample count packed in accum .a (scene alpha unused downstream).
// SES_ACCUM_CAP mirrors host kAccumCap (vk_render.ixx, bake -D).
import viz_util;

#ifndef SES_ACCUM_CAP
#define SES_ACCUM_CAP 512
#endif

[[vk::binding(0, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> scene;
[[vk::binding(1, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> accum;

struct Params {
    int accumulate;  // 0 = reset, 1 = keep averaging
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int2 p, size;
    if (oob(gid, accum, p, size)) {
        return;
    }
    float3 cur = scene[p].rgb;
    if (params.accumulate == 0) {
        accum[p] = float4(cur, 1.0);
        return;
    }
    float4 prev = accum[p];
    float n = min(prev.a + 1.0, float(SES_ACCUM_CAP));  // cap the window
    float3 avg = prev.rgb + (cur - prev.rgb) / n;
    accum[p] = float4(avg, n);
}
