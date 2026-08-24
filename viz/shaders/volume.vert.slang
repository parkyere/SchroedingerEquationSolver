// UBO (binding 0) shared with the fragment raymarch stage.
// mvp bakes the GL->Vulkan clip correction host-side.
import viz_ubo;

struct VSIn {
    [[vk::location(0)]] float3 pos : POSITION;
};
struct VSOut {
    float4 pos : SV_Position;
    [[vk::location(0)]] float3 v_world : TEXCOORD0;
};

[[vk::binding(0, 0)]] ConstantBuffer<Ubo, Std140DataLayout> ubo;

[shader("vertex")]
VSOut main(VSIn input) {
    VSOut o;
    o.pos = mul(ubo.mvp, float4(input.pos, 1.0));
    o.v_world = input.pos;
    return o;
}
