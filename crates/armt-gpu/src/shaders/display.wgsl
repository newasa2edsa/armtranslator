struct DisplayUniforms {
    fb_size: vec2<f32>,
    cursor_pos: vec2<f32>,
    cursor_size: f32,
    cursor_visible: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var fb_tex: texture_2d<f32>;
@group(0) @binding(1) var fb_sampler: sampler;
@group(0) @binding(2) var cursor_tex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> uni: DisplayUniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    // Full-screen triangle covering clip space (-1..1)
    let x = f32((vid & 1u) << 2u) - 1.0;
    let y = f32((vid & 2u) << 1u) - 1.0;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    // Flip Y for top-down guest framebuffer coordinates
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let base_sample = textureSample(fb_tex, fb_sampler, in.uv);
    var color = vec3<f32>(base_sample.rgb);

    if (uni.cursor_visible != 0u) {
        let px = in.uv * uni.fb_size;
        let rel = px - uni.cursor_pos;
        if (rel.x >= 0.0 && rel.y >= 0.0 && rel.x < uni.cursor_size && rel.y < uni.cursor_size) {
            let cuv = rel / uni.cursor_size;
            let c = textureSample(cursor_tex, fb_sampler, cuv);
            color = mix(color, c.rgb, c.a);
        }
    }

    return vec4<f32>(color, 1.0);
}
