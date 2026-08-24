// Colour/alpha + clip-side discard MIRROR ses::slice_shade /
// ses::clip_ray_interval (ses.cross_section, the tested source of truth).
import vec;
import viz_ubo;
import viz_util;

struct FSIn {
    float4 pos : SV_Position;
    [[vk::location(0)]] float3 v_world : TEXCOORD0;
};

[[vk::binding(0, 0)]] ConstantBuffer<Ubo, Std140DataLayout> ubo;

[[vk::binding(1, 0)]] Sampler3D<float4> psi_tex;
[[vk::binding(2, 0)]] Sampler1D<float4> phase_tex;

[shader("fragment")]
float4 main(FSIn input) : SV_Target {
    float3 v_world = input.v_world;
    if (ubo.clip.x > 0.5 &&
        ubo.clip.z * (comp(v_world, int(ubo.clip.y + 0.5)) - ubo.clip.w) > 0.0) {
        discard;
    }
    uint tw, th, td;
    psi_tex.GetDimensions(tw, th, td);
    float3 uvw = world_to_uvw(v_world, ubo.box_min.xyz, ubo.box_max.xyz,
                              float3(tw, th, td));
    float2 s = psi_tex.Sample(uvw).rg;
    float dens = dot(s, s) * ubo.inv_peak;      // |psi|^2
    float amp = sqrt(max(ubo.inv_peak, 0.0));

    int mode = int(ubo.slice.w + 0.5);
    float3 col;
    float bright;
    switch (mode) {
    case 1: {  // Re(psi)
        float r = clamp(s.r * amp, -1.0, 1.0);
        float3 pos = float3(1.0, 0.55, 0.15);
        float3 neg = float3(0.15, 0.45, 1.0);
        col = (r >= 0.0) ? lerp(float3(0.03, 0.03, 0.03), pos, r)
                         : lerp(float3(0.03, 0.03, 0.03), neg, -r);
        bright = abs(r);
        break;
    }
    case 2: {  // phase
        float ph = phase01(s);
        col = phase_tex.Sample(ph).rgb;
        bright = sqrt(clamp(dens, 0.0, 1.0));
        col *= 0.25 + 0.75 * bright;
        break;
    }
    default: {  // density (mode 0)
        float dd = clamp(dens, 0.0, 1.0);
        col = float3(pow(dd, 1.6), pow(dd, 0.9), 0.25 + 0.75 * dd);
        bright = dd;
        break;
    }
    }
    // base alpha keeps nodal planes visible
    float alpha = clamp(0.45 + 0.5 * bright, 0.0, 0.95);
    return float4(col * alpha, alpha);  // premultiplied
}
