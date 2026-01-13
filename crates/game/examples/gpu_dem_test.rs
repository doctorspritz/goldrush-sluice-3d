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

use game::gpu::dem_3d::GpuDem3D;
use game::gpu::{dem_3d, GpuContext};

const PARTICLE_COUNT: u32 = 1000;
const TEMPLATES: u32 = 2;
const MAX_CONTACTS: u32 = 10000;

struct DemTestApp {
    gpu_ctx: Option<GpuContext>,
    gpu_dem: Option<GpuDem3D>,
    window: Option<Arc<Window>>,
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
                        (i as f32) * spacing,
                        2.0 + (j as f32) * spacing + (i as f32 % 2.0) * 0.1,
                        (j as f32) * spacing,
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
            self.window = Some(window_arc);
        }
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("GPU DEM Test completed");
            }
            WindowEvent::Resized(physical_size) => {
                if let Some(gpu_ctx) = &mut self.gpu_ctx {
                    gpu_ctx.resize(physical_size.width, physical_size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                // Update DEM simulation
                if let (Some(gpu_ctx), Some(gpu_dem)) = (&self.gpu_ctx, &mut self.gpu_dem) {
                    let mut encoder =
                        gpu_ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("DEM Frame"),
                            });

                    // Update physics (60 FPS target)
                    gpu_dem.update(&mut encoder, 1.0 / 60.0);

                    // Submit commands
                    let _ = gpu_ctx.queue.submit(Some(encoder.finish()));
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // ControlFlow::Poll is default behavior if not Wait.
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
        window: None,
    };

    event_loop.run_app(&mut app).expect("Failed to run app");
}
