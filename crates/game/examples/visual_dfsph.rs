//! DFSPH Stability Test (Visual)
//!
//! Spawns a block of water particles and validates DFSPH stability visually.
//!
//! Run with: cargo run --example visual_dfsph --release

use bytemuck::{Pod, Zeroable};
use game::gpu::sph_dfsph::GpuSphDfsph;
use glam::{Mat4, Vec3};
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Parameters matching headless test for consistency
const H: f32 = 0.04;                   
const CELL_SIZE: f32 = H;
const SPACING: f32 = H / 2.0;             
const DT: f32 = 0.01;          
const SUB_STEPS: u32 = 1;            // 1 step * 0.01 = 0.01 (60Hz effective)
const GRID_SIZE: [u32; 3] = [32, 2048, 32]; 
const MAX_PARTICLES: u32 = 512_000;   

// Rendering
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _pad: f32,
}

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,

    particle_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,

    line_pipeline: wgpu::RenderPipeline,
    bucket_vertices: wgpu::Buffer,
    bucket_vertex_count: u32,

    _depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    sph: Option<GpuSphDfsph>,

    frame: u32,
    paused: bool,
    _start_time: Instant,
    last_fps_time: Instant,
    fps_frame_count: u32,
    current_fps: f32,

    camera_angle: f32,
    camera_pitch: f32,
    camera_distance: f32,
    
    spawned: bool,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            sph: None,
            frame: 0,
            paused: false,
            _start_time: Instant::now(),
            last_fps_time: Instant::now(),
            fps_frame_count: 0,
            current_fps: 0.0,
            camera_angle: 0.5,
            camera_pitch: 0.4,
            camera_distance: 3.0, 
            spawned: false,
        }
    }

    fn spawn_initial_block(&mut self) {
        if self.spawned { return; }
        
        let Some(ref gpu) = self.gpu else { return };
        let Some(ref mut sph) = self.sph else { return };

        let dim_x = 32u32;
        let dim_y = 500u32;
        let dim_z = 32u32;

        let block_width_x = dim_x as f32 * SPACING;
        let block_width_z = dim_z as f32 * SPACING;
        
        let offset_x = (GRID_SIZE[0] as f32 * CELL_SIZE - block_width_x) / 2.0;
        let offset_y = 0.01;
        let offset_z = (GRID_SIZE[2] as f32 * CELL_SIZE - block_width_z) / 2.0;

        let mut positions = Vec::with_capacity((dim_x * dim_y * dim_z) as usize);
        let mut velocities = Vec::with_capacity((dim_x * dim_y * dim_z) as usize);

        for x in 0..dim_x {
            for y in 0..dim_y {
                for z in 0..dim_z {
                     let pos = Vec3::new(
                        offset_x + x as f32 * SPACING + 0.5 * SPACING,
                        offset_y + y as f32 * SPACING + 0.5 * SPACING,
                        offset_z + z as f32 * SPACING + 0.5 * SPACING,
                    );
                    positions.push(pos);
                    velocities.push(Vec3::ZERO);
                }
            }
        }

        sph.upload_particles(&gpu.queue, &positions, &velocities);
        
        self.spawned = true;
        println!("Spawned {} particles", positions.len());
    }

    fn update_simulation(&mut self) {
        if self.paused { return; }

        if !self.spawned {
             self.spawn_initial_block();
        }

        let Some(ref gpu) = self.gpu else { return };
        let Some(ref mut sph) = self.sph else { return };

        for _ in 0..SUB_STEPS {
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("DFSPH Step"),
            });
            sph.step(&mut encoder, &gpu.queue);
            gpu.queue.submit(std::iter::once(encoder.finish()));
        }
    }

    fn render(&mut self) {
        let Some(ref gpu) = self.gpu else { return };
        let Some(ref sph) = self.sph else { return };

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output.texture.create_view(&Default::default());

        if gpu.config.width == 0 || gpu.config.height == 0 { return; }
        let aspect = gpu.config.width as f32 / gpu.config.height as f32;
        
        let center = Vec3::new(
            GRID_SIZE[0] as f32 * CELL_SIZE / 2.0,
            GRID_SIZE[1] as f32 * CELL_SIZE / 4.0,
            GRID_SIZE[2] as f32 * CELL_SIZE / 2.0,
        );
        let eye = center + Vec3::new(
            self.camera_distance * self.camera_angle.cos() * self.camera_pitch.cos(),
            self.camera_distance * self.camera_pitch.sin(),
            self.camera_distance * self.camera_angle.sin() * self.camera_pitch.cos(),
        );
        
        let view_matrix = Mat4::look_at_rh(eye, center, Vec3::Y);
        let proj_matrix = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 0.01, 100.0);

        let uniforms = Uniforms {
            view_proj: (proj_matrix * view_matrix).to_cols_array_2d(),
            camera_pos: eye.to_array(),
            _pad: 0.0,
        };
        gpu.queue.write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.15, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gpu.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&gpu.line_pipeline);
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.bucket_vertices.slice(..));
            pass.draw(0..gpu.bucket_vertex_count, 0..1);

            pass.set_pipeline(&gpu.particle_pipeline);
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, sph.position_buffer().slice(..));
            pass.draw(0..4, 0..sph.particle_count());
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        self.fps_frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_time).as_secs_f32();
        if elapsed >= 1.0 {
            self.current_fps = self.fps_frame_count as f32 / elapsed;
            self.fps_frame_count = 0;
            self.last_fps_time = now;
            let metrics = sph.compute_metrics(&gpu.device, &gpu.queue);
            println!("FPS: {:.1} | Particles: {} | MaxV: {:.3} | AvgKE: {:.5} | Err: {:.1}%", 
                self.current_fps, sph.particle_count(), metrics.max_velocity, metrics.avg_kinetic_energy, 
                metrics.density_error_percent);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }
        
        let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title("DFSPH Stability Visualizer").with_inner_size(winit::dpi::LogicalSize::new(1280, 720))).unwrap());
        self.window = Some(window.clone());

        let gpu = pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let surface = instance.create_surface(window.clone()).unwrap();
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false }).await.unwrap();
            let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { 
                required_limits: wgpu::Limits { max_storage_buffers_per_shader_stage: 16, ..Default::default() },
                ..Default::default() 
            }, None).await.unwrap();
            
            let config = surface.get_default_config(&adapter, window.inner_size().width, window.inner_size().height).unwrap();
            surface.configure(&device, &config);

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(PARTICLE_SHADER)) });
            let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(LINE_SHADER)) });

            let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth"),
                size: wgpu::Extent3d { width: config.width, height: config.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let depth_view = depth_texture.create_view(&Default::default());

            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&Uniforms { view_proj: [[0.0;4];4], camera_pos: [0.0;3], _pad: 0.0 }), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST });
            let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }], label: None });
            let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { layout: &uniform_bind_group_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() }], label: None });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { bind_group_layouts: &[&uniform_bind_group_layout], push_constant_ranges: &[], label: None });
            
            let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None, layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[wgpu::VertexBufferLayout { array_stride: 16, step_mode: wgpu::VertexStepMode::Instance, attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 0, shader_location: 0 }] }], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), targets: &[Some(wgpu::ColorTargetState { format: config.format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
                multisample: Default::default(), multiview: None, cache: None,
            });

            let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None, layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState { module: &line_shader, entry_point: Some("vs_main"), buffers: &[wgpu::VertexBufferLayout { array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex, attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }] }], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState { module: &line_shader, entry_point: Some("fs_main"), targets: &[Some(config.format.into())], compilation_options: Default::default() }),
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState { format: DEPTH_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
                multisample: Default::default(), multiview: None, cache: None,
            });

            let min = [0.0, 0.0, 0.0];
            let max = [GRID_SIZE[0] as f32 * CELL_SIZE, GRID_SIZE[1] as f32 * CELL_SIZE, GRID_SIZE[2] as f32 * CELL_SIZE];
            let verts = vec![
                [min[0], min[1], min[2]], [max[0], min[1], min[2]], [max[0], min[1], min[2]], [max[0], min[1], max[2]], [max[0], min[1], max[2]], [min[0], min[1], max[2]], [min[0], min[1], max[2]], [min[0], min[1], min[2]],
                [min[0], max[1], min[2]], [max[0], max[1], min[2]], [max[0], max[1], min[2]], [max[0], max[1], max[2]], [max[0], max[1], max[2]], [min[0], max[1], max[2]], [min[0], max[1], max[2]], [min[0], max[1], min[2]],
                [min[0], min[1], min[2]], [min[0], max[1], min[2]], [max[0], min[1], min[2]], [max[0], max[1], min[2]], [max[0], min[1], max[2]], [max[0], max[1], max[2]], [min[0], min[1], max[2]], [min[0], max[1], max[2]],
            ];
            let bucket_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&verts), usage: wgpu::BufferUsages::VERTEX });

            GpuState { device, queue, surface, config, particle_pipeline, uniform_buffer, uniform_bind_group, line_pipeline, bucket_vertices, bucket_vertex_count: verts.len() as u32, _depth_texture: depth_texture, depth_view }
        });

        let mut sph = GpuSphDfsph::new(&gpu.device, GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2], H, MAX_PARTICLES, DT);
        
        // Calibration during initialization (Not in draw loop)
        println!("Performing initial calibration...");
        let dim = 10u32;
        let mut spawn_pos = Vec::new();
        for x in 0..dim { for y in 0..dim { for z in 0..dim {
            spawn_pos.push(Vec3::new(x as f32 * SPACING, y as f32 * SPACING, z as f32 * SPACING));
        }}}
        sph.upload_particles(&gpu.queue, &spawn_pos, &vec![Vec3::ZERO; spawn_pos.len()]);
        sph.calibrate_rest_density(&gpu.device, &gpu.queue);
        sph.upload_particles(&gpu.queue, &[], &[]); // Clear
        println!("Calibration complete.");

        self.gpu = Some(gpu);
        self.sph = Some(sph);
    }
    
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
         match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                     if let PhysicalKey::Code(KeyCode::Space) = event.physical_key { self.paused = !self.paused; }
                }
            },
            WindowEvent::RedrawRequested => {
                self.update_simulation();
                self.render();
                self.frame += 1;
                self.window.as_ref().unwrap().request_redraw();
            },
            _ => {}
        }
    }
}

const LINE_SHADER: &str = r#"
struct Uniforms { view_proj: mat4x4<f32>, camera_pos: vec3<f32>, _pad: f32, };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@vertex fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> { return uniforms.view_proj * vec4(position, 1.0); }
@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4(0.8, 0.8, 0.8, 1.0); }
"#;

const PARTICLE_SHADER: &str = r#"
struct Uniforms { view_proj: mat4x4<f32>, camera_pos: vec3<f32>, _pad: f32, };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
struct VertexOutput { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32>, };
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @location(0) pos: vec4<f32>) -> VertexOutput {
    let size = 0.01; 
    let offsets = array<vec2<f32>, 4>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    let offset = offsets[vi] * size;
    let view_dir = normalize(uniforms.camera_pos - pos.xyz);
    let right = normalize(cross(vec3(0.0, 1.0, 0.0), view_dir));
    let up = cross(view_dir, right);
    let world_pos = pos.xyz + right * offset.x + up * offset.y;
    return VertexOutput(uniforms.view_proj * vec4(world_pos, 1.0), offsets[vi] * 0.5 + 0.5);
}
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv - vec2(0.5));
    let alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    if (alpha < 0.01) { discard; }
    return vec4(0.2, 0.5, 0.9, alpha * 0.8);
}
"#;

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
