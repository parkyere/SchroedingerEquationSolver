// Two-center band-limited Coulomb (H2+ Ehrenfest rotor): the CPU builder's
// sampling verbatim (ses::regularized_coulomb_potential) -- every cell takes
// -Z (2/pi) Si(pi r/h) / r per nucleus (-2Z/h at r = 0). Float32 Si is good
// to 7.5e-7 absolute (5e-7 relative on V); sin/cos ride the AS prefactors
// f ~ 1/x, g ~ 1/x^2, so their large-argument error is damped, not amplified.
module two_center;

public struct TwoCenterParams {
    public uint n;
    public uint nx;
    public uint ny;
    public uint nz;
    public float4 box_min;  // xyz
    public float4 cell_h;   // xyz, w = h (cubic: x spacing)
    public float4 c1;       // xyz nucleus 1, w = Z
    public float4 c2;       // xyz nucleus 2, w unused
    public float4 misc;     // x unused, yzw = 1/(2h) per axis
};

public float3 tc_point(TwoCenterParams p, uint ix, uint iy, uint iz) {
    return p.box_min.xyz + float3(ix, iy, iz) * p.cell_h.xyz;
}

// Si(x): term-recursive series below 4, Abramowitz-Stegun 5.2.38/39 above
// (ses::sine_integral verbatim).
float tc_sine_integral(float x) {
    if (x <= 4.0) {
        float x2 = x * x;
        float term = x;
        float s = 0.0;
        for (int n = 0; n < 20; ++n) {
            s += term / float(2 * n + 1);
            term *= -x2 / float((2 * n + 2) * (2 * n + 3));
        }
        return s;
    }
    float x2 = x * x;
    float fn = (((x2 + 38.027264) * x2 + 265.187033) * x2 + 335.677320) * x2 + 38.102495;
    float fd = (((x2 + 40.021433) * x2 + 322.624911) * x2 + 570.236280) * x2 + 157.105423;
    float gn = (((x2 + 42.242855) * x2 + 302.757865) * x2 + 352.018498) * x2 + 21.821899;
    float gd = (((x2 + 48.196927) * x2 + 482.485984) * x2 + 1114.978885) * x2 + 449.690326;
    float f = fn / fd / x;
    float g = gn / gd / x2;
    return 1.5707963267948966 - f * cos(x) - g * sin(x);
}

// (2/pi) Si(pi r/h) / r, unit charge (ses::band_limited_coulomb verbatim).
public float tc_band_limited(float r, float h) {
    return r > 0.0 ? 0.63661977236758134 * tc_sine_integral(3.14159265358979323 / h * r) / r
                   : 2.0 / h;
}

public float tc_coulomb(float3 r, float3 c, float z, float h) {
    return -z * tc_band_limited(length(r - c), h);
}

// V_1(r) and V_2(r) at one sample point.
public float2 tc_pair(TwoCenterParams p, float3 r) {
    float z = p.c1.w;
    float h = p.cell_h.w;
    return float2(tc_coulomb(r, p.c1.xyz, z, h), tc_coulomb(r, p.c2.xyz, z, h));
}
