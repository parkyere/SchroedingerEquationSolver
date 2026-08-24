// Premultiplied-alpha passthrough; consume with premultiplied blend.
// Baked twice (flow_frag + overlay_frag): one source, two pipeline blobs.
struct FSIn {
    float4 pos : SV_Position;
    [[vk::location(0)]] float4 v_color : TEXCOORD0;
};

[shader("fragment")]
float4 main(FSIn input) : SV_Target {
    return input.v_color;
}
