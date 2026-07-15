// complex.slang -- zero-overhead complex-scalar type shared by the ses_vk
// kernels. A Complex IS a bare float2 (re, im); every operator inlines to the
// underlying float ops. Import with `import complex;`.
module complex;

public struct Complex {
    public float2 v;  // (re, im)

    public __init(float2 f) { v = f; }
    public __init(float re, float im) { v = float2(re, im); }

    public property float re {
        get { return v.x; }
        set { v.x = newValue; }
    }
    public property float im {
        get { return v.y; }
        set { v.y = newValue; }
    }
}

// z1 +/- z2, componentwise.
public Complex operator+(Complex a, Complex b) { return Complex(a.v + b.v); }
public Complex operator-(Complex a, Complex b) { return Complex(a.v - b.v); }

// z1 * z2 = (ac-bd, ad+bc).
public Complex operator*(Complex a, Complex b) {
    return Complex(a.v.x * b.v.x - a.v.y * b.v.y,
                   a.v.x * b.v.y + a.v.y * b.v.x);
}

// Real scale (both orders) and real divide.
public Complex operator*(Complex a, float s) { return Complex(a.v * s); }
public Complex operator*(float s, Complex a) { return Complex(a.v * s); }
public Complex operator/(Complex a, float s) { return Complex(a.v / s); }

public Complex conj(Complex a) { return Complex(a.v.x, -a.v.y); }
public float norm2(Complex a) { return dot(a.v, a.v); }         // |z|^2
public float arg(Complex a) { return atan2(a.v.y, a.v.x); }     // phase in (-pi, pi]

// Componentwise lerp of the re/im pair.
public Complex mix(Complex a, Complex b, float t) { return Complex(lerp(a.v, b.v, t)); }

// Hermitian product conj(a)*b = (a.x b.x + a.y b.y, a.x b.y - a.y b.x). Term
// order is fixed so FMA contraction stays bit-reproducible.
public Complex mul_conj(Complex a, Complex b) {
    return Complex(a.v.x * b.v.x + a.v.y * b.v.y,
                   a.v.x * b.v.y - a.v.y * b.v.x);
}
public Complex cis(float theta) { return Complex(cos(theta), sin(theta)); }  // e^{i theta}
