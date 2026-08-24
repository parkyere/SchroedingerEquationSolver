// main() mirrors ses::slice_quad_corner (ses.cross_section); Ubo shared with the
// volume renderer. gl_VertexIndex avoids the shaderDrawParameters capability that
// SV_VertexID needs.
import glsl;
import vec;
import viz_ubo;

[[vk::binding(0, 0)]] ConstantBuffer<Ubo, Std140DataLayout> ubo;

struct VSOut {
    float4 pos : SV_Position;
    [[vk::location(0)]] float3 v_world : TEXCOORD0;
};

// CCW winding.
static const float2 kST[6] = {
    float2(0, 0), float2(1, 0), float2(1, 1),
    float2(0, 0), float2(1, 1), float2(0, 1)
};

[shader("vertex")]
VSOut main() {
    VSOut o;
    uint vid = uint(gl_VertexIndex);
    int a = int(ubo.slice.y + 0.5);
    int u = (a + 1) % 3;
    int w = (a + 2) % 3;
    float2 st = kST[vid];
    float3 p = float3(0.0, 0.0, 0.0);
    set_comp(p, a, ubo.slice.z);
    set_comp(p, u, lerp(comp(ubo.box_min.xyz, u), comp(ubo.box_max.xyz, u), st.x));
    set_comp(p, w, lerp(comp(ubo.box_min.xyz, w), comp(ubo.box_max.xyz, w), st.y));
    o.v_world = p;
    o.pos = mul(ubo.mvp, float4(p, 1.0));
    return o;
}
