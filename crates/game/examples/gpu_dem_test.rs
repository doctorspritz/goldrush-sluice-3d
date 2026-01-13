//! GPU DEM Test: Simple particle collision and settling
//!
//! Tests basic GPU DEM functionality with simple falling spheres.
//! Verifies spatial hashing, collision detection, and integration.
//!
//! Run with: cargo run --example gpu_dem_test --release

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use game::gpu::dem_render::DemRenderer;
use game::gpu::dem_3d::GpuDem3D;
use game::gpu::GpuContext;

const PARTICLE_COUNT: u32 = 1000;
const TEMPLATES: u32 = 2;
const MAX_CONTACTS: u32 = 30000;

struct DemTestApp {
    gpu_ctx: Option<GpuContext>,
    gpu_dem: Option<GpuDem3D>,
    renderer: Option<DemRenderer>,
    depth_view: Option<wgpu::TextureView>,
    window: Option<Arc<Window>>,
}

impl DemTestApp {
    fn create_depth_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

impl ApplicationHandler for DemTestApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu_ctx.is_none() {
            let window = event_loop
                .create_window(winit::window::WindowAttributes::default())
                .expect("Failed to create window");

            let window_arc = Arc::new(window);

            // Initialize GPU
            let gpu_ctx = pollster::block_on(GpuContext::new(window_arc.clone()));

            // Create GPU DEM
            let mut gpu_dem = GpuDem3D::new(
                gpu_ctx.device.clone(),
                gpu_ctx.queue.clone(),
                PARTICLE_COUNT,
                TEMPLATES,
                MAX_CONTACTS,
            );
            
            // Create Renderer
            let renderer = DemRenderer::new(&gpu_ctx.device, gpu_ctx.config.format);
            let depth_view = Self::create_depth_texture(&gpu_ctx.device, gpu_ctx.config.width, gpu_ctx.config.height);

            // Create test templates
            let template_small = sim3d::ClumpTemplate3D::generate(
                sim3d::ClumpShape3D::Tetra,
                0.005,  // 5mm radius
                0.0001, // 100g
            );
            let template_large = sim3d::ClumpTemplate3D::generate(
                sim3d::ClumpShape3D::Cube2,
                0.01,  // 1cm radius
                0.001, // 1kg
            );

            let template_id_small = gpu_dem.add_template(template_small);
            let template_id_large = gpu_dem.add_template(template_large);

            // Spawn particles in a grid
            let spacing = 0.05; // 5cm spacing
            let per_side = (PARTICLE_COUNT as f32).sqrt() as u32;

            for i in 0..per_side {
                for j in 0..per_side {
                    let idx = i * per_side + j;
                    if idx >= PARTICLE_COUNT {
                        break;
                    }

                    let position = Vec3::new(
                        (i as f32) * spacing - 1.0,
                        2.0 + (j as f32) * spacing + (i as f32 % 2.0) * 0.1,
                        (j as f32) * spacing - 1.0,
                    );

                    let template_id = if (idx % 3) == 0 {
                        template_id_small
                    } else {
                        template_id_large
                    };
                    gpu_dem.spawn_clump(template_id, position, Vec3::ZERO);
                }
            }

            self.gpu_ctx = Some(gpu_ctx);
            self.gpu_dem = Some(gpu_dem);
            self.renderer = Some(renderer);
            self.depth_view = Some(depth_view);
            self.window = Some(window_arc);
        }
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("GPU DEM Test completed");
                std::process::exit(0);
            }
            WindowEvent::Resized(physical_size) => {
                if let Some(gpu_ctx) = &mut self.gpu_ctx {
                    gpu_ctx.resize(physical_size.width, physical_size.height);
                    // Recreate depth texture
                    self.depth_view = Some(Self::create_depth_texture(&gpu_ctx.device, physical_size.width, physical_size.height));
                }
            }
            WindowEvent::RedrawRequested => {
                // Update DEM simulation
                if let (Some(gpu_ctx), Some(gpu_dem), Some(renderer), Some(depth_view)) = (
                    &self.gpu_ctx, 
                    &mut self.gpu_dem, 
                    &self.renderer,
                    &self.depth_view
                ) {
                    let mut encoder =
                        gpu_ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("DEM Frame"),
                            });

                    // Update physics (60 FPS target)
                    gpu_dem.update(&mut encoder, 1.0 / 60.0);
                    
                    // Render
                    let frame = match gpu_ctx.surface.get_current_texture() {
                        Ok(frame) => frame,
                        Err(wgpu::SurfaceError::Outdated) => return,
                        Err(e) => {
                            eprintln!("Surface error: {:?}", e);
                            return;
                        }
                    };
                    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                    
                    // Clear pass
                    {
                        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Clear Pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.1, g: 0.1, b: 0.15, a: 1.0 }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: depth_view,
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(1.0),
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });
                    }
                    
                    // Setup Camera
                    let aspect = gpu_ctx.config.width as f32 / gpu_ctx.config.height as f32;
                    let proj = glam::Mat4::perspective_rh(45.0f32.to_radians(), aspect, 0.1, 100.0);
                    let camera_pos = Vec3::new(3.0, 3.0, 3.0);
                    let target = Vec3::new(0.0, 0.0, 0.0);
                    let view_mat = glam::Mat4::look_at_rh(camera_pos, target, Vec3::Y);

                    renderer.render(
                        &gpu_ctx.device,
                        &gpu_ctx.queue,
                        &mut encoder,
                        &view,
                        depth_view,
                        gpu_dem,
                        view_mat.to_cols_array_2d(),
                        proj.to_cols_array_2d(),
                        camera_pos.to_array(),
                    );

                    // Submit commands
                    gpu_ctx.queue.submit(Some(encoder.finish()));
                    frame.present();
                    
                    // Request next frame
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    println!("GPU DEM Test");
    println!("===============");
    println!();
    println!("Initializing GPU DEM with {} particles...", PARTICLE_COUNT);

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = DemTestApp {
        gpu_ctx: None,
        gpu_dem: None,
        renderer: None,
        depth_view: None,
        window: None,
    };

    event_loop.run_app(&mut app).expect("Failed to run app");
}
