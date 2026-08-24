// Std140 mirror of host VolumeUbo (viz/src/vk_render.ixx): field order = byte
// contract. Every viz UBO descriptor spans the full struct, so all stages
// declare the whole block. SES_MAX_MARKERS mirrors host kMaxMarkers (bake -D).
module viz_ubo;

#ifndef SES_MAX_MARKERS
#define SES_MAX_MARKERS 64
#endif

public struct Ubo {
    public float4x4 mvp;
    public float4 eye;
    public float4 box_min;
    public float4 box_max;
    public float4 proton_center;
    public float4 proton_color;
    public float inv_peak;
    public float absorbance;
    public float proton_radius;
    public float jitter_frame;
    public float4 clip;     // .x enable, .y axis(0/1/2), .z sign(+-1), .w offset
    public float4 slice;    // .x enable, .y axis, .z offset, .w map mode
    public float4 barrier;  // .x enable, .y x_lo, .z x_hi, .w fog density
    public float4 marker_cr[SES_MAX_MARKERS];   // xyz center, w radius
    public float4 marker_col[SES_MAX_MARKERS];  // rgb
    public float4 marker_meta;                  // .x count
};
