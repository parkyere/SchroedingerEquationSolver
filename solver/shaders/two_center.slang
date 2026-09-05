// Two-center finite-volume Coulomb (H2+ Ehrenfest rotor): the CPU builder's
// sampling verbatim (ses::regularized_coulomb_potential) -- cells within the
// averaging radius take the cube average of -Z/|r-c| (Waldvogel closed form,
// the nucleus cell included), an exact hit the tabulated -Z*C/h, the rest
// -Z/r. Float32: the log terms cancel to ~1e-5 relative inside 3 cells
// (vkcheck bounds the effect on psi at 2e-6 over 20 steps).
module two_center;

public struct TwoCenterParams {
    public uint n;
    public uint nx;
    public uint ny;
    public uint nz;
    public float4 box_min;  // xyz
    public float4 cell_h;   // xyz, w = h (cubic: x spacing)
    public float4 c1;       // xyz nucleus 1, w = Z
    public float4 c2;       // xyz nucleus 2, w = center_v (-Z*C/h, exact hit)
    public float4 misc;     // x = averaging radius, yzw = 1/(2h) per axis
};

public float3 tc_point(TwoCenterParams p, uint ix, uint iy, uint iz) {
    return p.box_min.xyz + float3(ix, iy, iz) * p.cell_h.xyz;
}

// Waldvogel antiderivative, d^3F/dxdydz = 1/r: principal-value atan (atan2's
// pi shift does not cancel over the corners); a zero coordinate takes the
// limit (x*y*ln -> 0, x^2 atan -> 0); ln(a + r), a < 0, via s2/(r - a).
float tc_corner(float x, float y, float z) {
    float r = sqrt(x * x + y * y + z * z);
    float f = 0.0;
    if (x * y != 0.0) {
        f += x * y * log(z >= 0.0 ? z + r : (x * x + y * y) / (r - z));
    }
    if (y * z != 0.0) {
        f += y * z * log(x >= 0.0 ? x + r : (y * y + z * z) / (r - x));
    }
    if (z * x != 0.0) {
        f += z * x * log(y >= 0.0 ? y + r : (z * z + x * x) / (r - y));
    }
    if (x != 0.0) {
        f -= 0.5 * x * x * atan(y * z / (x * r));
    }
    if (y != 0.0) {
        f -= 0.5 * y * y * atan(z * x / (y * r));
    }
    if (z != 0.0) {
        f -= 0.5 * z * z * atan(x * y / (z * r));
    }
    return f;
}

// (1/h^3) integral over the h cube centered at the origin of 1/|r - d|:
// signed corner sum, (-1)^(number of lower limits).
public float tc_cube_average(float3 d, float h) {
    float3 lo = -0.5 * h - d;
    float3 hi = 0.5 * h - d;
    float acc = 0.0;
    for (int m = 0; m < 8; ++m) {
        bool ux = (m & 1) != 0;
        bool uy = (m & 2) != 0;
        bool uz = (m & 4) != 0;
        int lower = (ux ? 0 : 1) + (uy ? 0 : 1) + (uz ? 0 : 1);
        float f = tc_corner(ux ? hi.x : lo.x, uy ? hi.y : lo.y, uz ? hi.z : lo.z);
        acc += (lower % 2 == 0) ? f : -f;
    }
    return acc / (h * h * h);
}

public float tc_coulomb(float3 r, float3 c, float z, float center_v, float h,
                        float r_avg) {
    float3 d = r - c;
    float dist = length(d);
    if (dist == 0.0) {
        return center_v;
    }
    if (dist < r_avg) {
        return -z * tc_cube_average(d, h);
    }
    return -z / dist;
}

// V_1(r) and V_2(r) at one sample point.
public float2 tc_pair(TwoCenterParams p, float3 r) {
    float z = p.c1.w;
    float cv = p.c2.w;
    float h = p.cell_h.w;
    float r_avg = p.misc.x;
    return float2(tc_coulomb(r, p.c1.xyz, z, cv, h, r_avg),
                  tc_coulomb(r, p.c2.xyz, z, cv, h, r_avg));
}
