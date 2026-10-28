// Read-only physical events. Every record is vec4: old h / limited X / Z /
// boundary volume, or pre/post-scour geometry / precision / C / water depth.
struct Event { cells:u32, offset:u32, bed:u32, pad:u32 }
@group(0) @binding(0) var<storage,read> depth:array<f32>;
@group(0) @binding(1) var<storage,read> flux_x:array<f32>;
@group(0) @binding(2) var<storage,read> flux_z:array<f32>;
@group(0) @binding(3) var<storage,read> aux:array<f32>;
@group(0) @binding(4) var<storage,read> concentration:array<f32>;
@group(0) @binding(5) var<storage,read> paydirt:array<f32>;
@group(0) @binding(6) var<storage,read> bed_precision:array<vec4<f32>>;
@group(1) @binding(0) var<storage,read_write> events:array<vec4<f32>>;
@group(1) @binding(1) var<uniform> event:Event;
@compute @workgroup_size(64)
fn record_event(@builtin(global_invocation_id) g:vec3<u32>) {
    let i=g.x;if (i>=event.cells) {return;}
    if (event.bed==0u) {
        events[event.offset+i]=vec4(depth[i],flux_x[i],flux_z[i],aux[event.cells+i*12u+2u]);
    } else {
        events[event.offset+i]=vec4(paydirt[i],bed_precision[i].w,concentration[i],depth[i]);
    }
}
