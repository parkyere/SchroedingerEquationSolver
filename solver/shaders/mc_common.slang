// Marching-cubes shared tables + classification (mc_classify / mc_emit must
// agree or the indirect draw silently breaks). Tables are verbatim ports of
// the CPU core ses.mc.tables.
module mc_common;

// mc_tab flat layout contract; host serialization names the same numbers.
public static const int kMcEdgeOff = 0;      // edge masks   [0, 256)
public static const int kMcTriOff = 256;     // tris 256x16  [256, 4352)
public static const int kMcTriStride = 16;
public static const int kMcCountOff = 4352;  // tri counts   [4352, 4608)

public static const int3 kMcCorner[8] = {
    int3(0, 0, 0), int3(1, 0, 0), int3(1, 1, 0), int3(0, 1, 0),
    int3(0, 0, 1), int3(1, 0, 1), int3(1, 1, 1), int3(0, 1, 1)
};
public static const int2 kMcEdge[12] = {
    int2(0, 1), int2(1, 2), int2(2, 3), int2(3, 0),
    int2(4, 5), int2(5, 6), int2(6, 7), int2(7, 4),
    int2(0, 4), int2(1, 5), int2(2, 6), int2(3, 7)
};

public int mc_flat_at(int3 p, int3 npts) {
    return p.x + npts.x * (p.y + npts.y * p.z);
}

// corner c below iso -> bit c
public int mc_classify_cube(StructuredBuffer<float, Std430DataLayout> den,
                            int3 cell, int3 npts, float iso) {
    int cube = 0;
    for (int c = 0; c < 8; ++c) {
        if (den[mc_flat_at(cell + kMcCorner[c], npts)] < iso) {
            cube |= 1 << c;
        }
    }
    return cube;
}
