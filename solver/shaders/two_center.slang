// Two-center regularized Coulomb (H2+ Ehrenfest rotor): the CPU builder's
// sampling verbatim (ses::regularized_coulomb_potential) -- a cell that lands
// on a nucleus takes the cell-average value, everything else -Z/r.
module two_center;

public struct TwoCenterParams {
    public uint n;
    public uint nx;
    public uint ny;
    public uint nz;
    public float4 box_min;  // xyz
    public float4 cell_h;   // xyz
    public float4 c1;       // xyz nucleus 1, w = Z
    public float4 c2;       // xyz nucleus 2, w = center_v (-Z*kCellAvg/h)
    public float4 misc;     // x = coincidence eps, yzw = 1/(2h) per axis
};

public float3 tc_point(TwoCenterParams p, uint ix, uint iy, uint iz) {
    return p.box_min.xyz + float3(ix, iy, iz) * p.cell_h.xyz;
}

public float tc_coulomb(float3 r, float3 c, float z, float center_v, float eps) {
    float d = length(r - c);
    return d < eps ? center_v : -z / d;
}

// V_1(r) and V_2(r) at one sample point.
public float2 tc_pair(TwoCenterParams p, float3 r) {
    float z = p.c1.w;
    float cv = p.c2.w;
    float eps = p.misc.x;
    return float2(tc_coulomb(r, p.c1.xyz, z, cv, eps),
                  tc_coulomb(r, p.c2.xyz, z, cv, eps));
}
