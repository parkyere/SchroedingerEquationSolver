// Radial ladder r_i = (i+1)*h: origin pin (u(0)=0, l=0 limit u_0/h) + linear
// interp with u(rmax)=0 outer pad. Shared by synth / mcwf_axpy /
// project_deposit.
module radial_sample;

public struct RadialBin {
    public bool origin;  // r < h
    public int i0;
    public float frac;
};

public RadialBin radial_bin(float r, float h) {
    RadialBin b;
    b.origin = (r < h);
    b.i0 = 0;
    b.frac = 0.0;
    if (!b.origin) {
        float t = r / h - 1.0;
        b.i0 = int(t);
        b.frac = t - float(b.i0);
    }
    return b;
}

// u(r)/r for one n_radial slot at base; caller guards r < rmax (i0 in range).
public float radial_u_over_r(StructuredBuffer<float, Std430DataLayout> u,
                             int base, RadialBin b, float r, float h,
                             int n_radial) {
    if (b.origin) {
        return u[base] / h;
    }
    float ua = u[base + b.i0];
    float ub = (b.i0 + 1 < n_radial) ? u[base + b.i0 + 1] : 0.0;
    return ((1.0 - b.frac) * ua + b.frac * ub) / r;
}
