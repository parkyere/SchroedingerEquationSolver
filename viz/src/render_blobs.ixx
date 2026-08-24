module;
#include <mesh_vert_spv.h>
#include <mesh_frag_spv.h>
#include <marker_frag_spv.h>
#include <volume_vert_spv.h>
#include <volume_frag_spv.h>
#include <slice_vert_spv.h>
#include <slice_frag_spv.h>
#include <accum_spv.h>
#include <bloom_down_spv.h>
#include <bloom_up_spv.h>
#include <compose_spv.h>
#include <particles_spv.h>
#include <occupancy_spv.h>
#include <occ_dilate_spv.h>
#include <shadow_spv.h>
#include <flow_vert_spv.h>
#include <flow_frag_spv.h>
#include <overlay_vert_spv.h>
#include <overlay_frag_spv.h>
export module ses.vk.render_blobs;
export import ses.vk.render;


// Baked offline from viz/shaders/.


export namespace ses_vk {

inline RenderKernels render_blobs() {
    RenderKernels r;
    r.mesh_vert = {k_mesh_vert_spv, k_mesh_vert_spv_size};
    r.mesh_frag = {k_mesh_frag_spv, k_mesh_frag_spv_size};
    r.marker_frag = {k_marker_frag_spv, k_marker_frag_spv_size};
    r.volume_vert = {k_volume_vert_spv, k_volume_vert_spv_size};
    r.volume_frag = {k_volume_frag_spv, k_volume_frag_spv_size};
    r.slice_vert = {k_slice_vert_spv, k_slice_vert_spv_size};
    r.slice_frag = {k_slice_frag_spv, k_slice_frag_spv_size};
    r.accum = {k_accum_spv, k_accum_spv_size};
    r.bloom_down = {k_bloom_down_spv, k_bloom_down_spv_size};
    r.bloom_up = {k_bloom_up_spv, k_bloom_up_spv_size};
    r.compose = {k_compose_spv, k_compose_spv_size};
    r.particles = {k_particles_spv, k_particles_spv_size};
    r.occupancy = {k_occupancy_spv, k_occupancy_spv_size};
    r.occ_dilate = {k_occ_dilate_spv, k_occ_dilate_spv_size};
    r.shadow = {k_shadow_spv, k_shadow_spv_size};
    r.flow_vert = {k_flow_vert_spv, k_flow_vert_spv_size};
    r.flow_frag = {k_flow_frag_spv, k_flow_frag_spv_size};
    r.overlay_vert = {k_overlay_vert_spv, k_overlay_vert_spv_size};
    r.overlay_frag = {k_overlay_frag_spv, k_overlay_frag_spv_size};
    return r;
}

}  // namespace ses_vk
