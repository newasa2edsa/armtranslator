use crate::adapter::{create_gpu_instance, select_best_adapter, GpuAdapterInfo};
use armt_core::{ArmtError, Result};
use armt_hw::Framebuffer;
use bytemuck::{Pod, Zeroable};
use eframe::egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use eframe::epaint::PaintCallbackInfo;
use eframe::wgpu;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DisplayUniforms {
    fb_size: [f32; 2],
    cursor_pos: [f32; 2],
    cursor_size: f32,
    cursor_visible: u32,
    _pad0: u32,
    _pad1: u32,
}

#[derive(Default)]
pub struct GpuRenderStats {
    pub uploads: AtomicU64,
    pub last_upload_us: AtomicU64,
}

/// GPU-side pipeline and VRAM textures for presenting the ARM64 guest framebuffer + hardware cursor.
pub struct GuestDisplayGpuPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    fb_texture: wgpu::Texture,
    cursor_texture: wgpu::Texture,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    fb_width: u32,
    fb_height: u32,
    last_generation: u64,
    staging_bgra: Vec<u8>,
    target_format: wgpu::TextureFormat,
}

impl GuestDisplayGpuPipeline {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
        fb: &Framebuffer,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("armtranslator_display_wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/display.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("display_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("display_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("display_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let fb_width = fb.width.max(1);
        let fb_height = fb.height.max(1);
        let fb_texture = create_scanout_texture(device, fb_width, fb_height, target_format.is_srgb());
        let cursor_texture = create_cursor_texture(device, queue, fb, target_format.is_srgb());

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("display_uniforms"),
            size: std::mem::size_of::<DisplayUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = create_bind_group(
            device,
            &bind_group_layout,
            &fb_texture,
            &sampler,
            &cursor_texture,
            &uniform_buffer,
        );

        let mut this = Self {
            pipeline,
            bind_group_layout,
            sampler,
            fb_texture,
            cursor_texture,
            uniform_buffer,
            bind_group,
            fb_width,
            fb_height,
            last_generation: 0,
            staging_bgra: Vec::new(),
            target_format,
        };
        this.sync_from_framebuffer(device, queue, fb, None);
        this
    }

    pub fn sync_from_framebuffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        fb: &Framebuffer,
        stats: Option<&GpuRenderStats>,
    ) {
        if fb.width != self.fb_width || fb.height != self.fb_height {
            self.fb_width = fb.width.max(1);
            self.fb_height = fb.height.max(1);
            self.fb_texture = create_scanout_texture(
                device,
                self.fb_width,
                self.fb_height,
                self.target_format.is_srgb(),
            );
            self.bind_group = create_bind_group(
                device,
                &self.bind_group_layout,
                &self.fb_texture,
                &self.sampler,
                &self.cursor_texture,
                &self.uniform_buffer,
            );
            self.last_generation = 0;
        }

        let gen = fb.generation();
        if gen != self.last_generation {
            let t0 = Instant::now();
            fb.copy_bgra_bytes(&mut self.staging_bgra);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.fb_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &self.staging_bgra,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.fb_width * 4),
                    rows_per_image: Some(self.fb_height),
                },
                wgpu::Extent3d {
                    width: self.fb_width,
                    height: self.fb_height,
                    depth_or_array_layers: 1,
                },
            );
            self.last_generation = gen;
            if let Some(st) = stats {
                st.uploads.fetch_add(1, Ordering::Relaxed);
                st.last_upload_us
                    .store(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        }

        let (cx, cy, vis) = fb.cursor_state();
        let uniforms = DisplayUniforms {
            fb_size: [self.fb_width as f32, self.fb_height as f32],
            cursor_pos: [cx as f32, cy as f32],
            cursor_size: 32.0,
            cursor_visible: if vis { 1 } else { 0 },
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn render_into_pass<'r>(&self, pass: &mut wgpu::RenderPass<'r>) {
        let _ = self.target_format;
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn create_scanout_texture(device: &wgpu::Device, w: u32, h: u32, srgb: bool) -> wgpu::Texture {
    let format = if srgb {
        wgpu::TextureFormat::Bgra8UnormSrgb
    } else {
        wgpu::TextureFormat::Bgra8Unorm
    };
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("guest_scanout_bgra"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn create_cursor_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fb: &Framebuffer,
    srgb: bool,
) -> wgpu::Texture {
    let format = if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("guest_cursor_rgba"),
        size: wgpu::Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let cursor = fb.cursor_rgba.lock().unwrap();
    let mut bytes = Vec::with_capacity(32 * 32 * 4);
    for &px in cursor.iter().take(32 * 32) {
        bytes.extend_from_slice(&px.to_le_bytes());
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(32 * 4),
            rows_per_image: Some(32),
        },
        wgpu::Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 1,
        },
    );
    tex
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    fb_tex: &wgpu::Texture,
    sampler: &wgpu::Sampler,
    cursor_tex: &wgpu::Texture,
    uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let fb_view = fb_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let cursor_view = cursor_tex.create_view(&wgpu::TextureViewDescriptor::default());
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("display_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&fb_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&cursor_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    })
}

/// Custom `egui_wgpu` paint callback executing `display.wgsl` directly on the NVIDIA RTX GPU.
pub struct DisplayPaintCallback {
    pub fb: Framebuffer,
    pub stats: Arc<GpuRenderStats>,
}

impl CallbackTrait for DisplayPaintCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(pipe) = callback_resources.get_mut::<GuestDisplayGpuPipeline>() {
            pipe.sync_from_framebuffer(device, queue, &self.fb, Some(&self.stats));
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        if let Some(pipe) = callback_resources.get::<GuestDisplayGpuPipeline>() {
            pipe.render_into_pass(render_pass);
        }
    }
}

/// Execute a real headless GPU shader self-test on the NVIDIA RTX adapter:
/// creates a `wgpu::Device`, uploads a test guest framebuffer + cursor, runs `display.wgsl`
/// into an offscreen render target, copies back via staging buffer, and verifies non-zero pixels.
pub fn run_headless_gpu_self_test() -> Result<(GpuAdapterInfo, u32)> {
    let instance = create_gpu_instance();
    let adapter = select_best_adapter(&instance, None)?;
    let info = GpuAdapterInfo::from_wgpu(adapter.get_info());

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("armtranslator_headless_rtx_test"),
        ..Default::default()
    }))
    .map_err(|e| ArmtError::Device(format!("wgpu request_device failed: {e}")))?;

    let w = 256u32;
    let h = 256u32;
    let fb = Framebuffer::new(w, h);
    fb.clear(0x001E_3A8A); // deep blue guest desktop
    fb.fill_rect(32, 32, 128, 64, 0x003B_82F6);
    fb.set_cursor_pos(40, 40, true);

    let format = wgpu::TextureFormat::Bgra8Unorm;
    let pipe = GuestDisplayGpuPipeline::new(&device, &queue, format, &fb);

    let out_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("offscreen_rt"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes_per_row = w * 4;
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen_readback"),
        size: (bytes_per_row * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("headless_rtx_encoder"),
    });
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("headless_rtx_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &out_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pipe.render_into_pass(&mut rpass);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &out_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .map_err(|_| ArmtError::Device("GPU readback channel closed".into()))?
        .map_err(|e| ArmtError::Device(format!("GPU buffer map failed: {e:?}")))?;

    let data = slice
        .get_mapped_range()
        .map_err(|e| ArmtError::Device(format!("GPU buffer map view failed: {e:?}")))?;
    let non_zero = data
        .chunks_exact(4)
        .filter(|c| c[0] != 0 || c[1] != 0 || c[2] != 0)
        .count() as u32;
    drop(data);
    readback_buf.unmap();

    if non_zero < w * h / 2 {
        return Err(ArmtError::Device(format!(
            "Headless GPU shader render produced too few pixels: {non_zero}"
        )));
    }
    Ok((info, non_zero))
}
