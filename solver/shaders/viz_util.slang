// Shared viz helpers; formulas verbatim from the former per-file copies.
module viz_util;

public static const float kPi = 3.14159265358979;
public static const float kTwoPi = 6.28318530717959;

// atan2 phase -> [0,1) (phase_tex colour-wheel contract).
public float phase01(float2 s) {
    return (atan2(s.g, s.r) + kPi) / kTwoPi;
}

// Interleaved gradient noise, golden-ratio rotated per frame.
public float ign(float2 pix, float frame) {
    float2 p = pix + 5.588238 * frame;
    return frac(52.9829189 * frac(0.06711056 * p.x + 0.00583715 * p.y));
}

// World -> uvw; half-texel: grid point i sits at texel center (i+0.5)/n,
// else the cloud is half a cell off.
public float3 world_to_uvw(float3 p, float3 box_min, float3 box_max,
                           float3 size) {
    return (p - box_min) / (box_max - box_min) + 0.5 / size;
}

// 2D dispatch guard; p/size valid when false.
public bool oob(uint3 gid, RWTexture2D<float4> tex, out int2 p,
                out int2 size) {
    uint w, h;
    tex.GetDimensions(w, h);
    size = int2(w, h);
    p = int2(gid.xy);
    return any(p >= size);
}
