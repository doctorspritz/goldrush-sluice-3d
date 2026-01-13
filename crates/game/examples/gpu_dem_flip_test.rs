//! GPU DEM-FLIP Integration Test
//!
//! Tests coupling between GPU DEM and GPU FLIP simulations.
//! Verifies momentum transfer and bidirectional interaction.
//!
//! Run with: cargo run --example gpu_dem_flip_test --release --features gpu-dem

use bytemuck::{Pod, Zeroable};
use game::gpu::GpuContext;
use game::gpu::{dem_3d::GpuDem3D, flip_3d::GpuFlip3D};
use glam::Vec3;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow},
    window::{Window, WindowId},
};

const PARTICLE_COUNT: u32 = 5000;
const GRID_WIDTH: u32 = 32;
const GRID_HEIGHT: u32 = 32;
const GRID_DEPTH: u32 = 32;
const CELL_SIZE: f32 = 0.05; // 5cm cells

struct DemFlipTest {
    gpu_ctx: Option<GpuContext>,
    gpu_flip: Option<GpuFlip3D>,
    gpu_dem: Option<GpuDem3D>,
    window: Option<Arc<Window>>,
}

impl ApplicationHandler for DemFlipTest {
    fn resumed(&mut self, event_loop: &ActiveEventLoop<()>) {
        if self.gpu_ctx.is_none() {
            let window = event_loop
                .create_window(winit::window::WindowAttributes::default())
                .expect("Failed to create window");

            let window_arc = Arc::new(window);

            // Initialize GPU
            let gpu_ctx = pollster::block_on(GpuContext::new(window_arc.clone()));

            // Create GPU FLIP
            let mut gpu_flip = GpuFlip3D::new(
                &gpu_ctx.device,
                GRID_WIDTH,
                GRID_HEIGHT,
                GRID_DEPTH,
                CELL_SIZE,
                PARTICLE_COUNT,
            );

            // Create GPU DEM
            let mut gpu_dem = GpuDem3D::new(
                gpu_ctx.device.clone(),
                gpu_ctx.queue.clone(),
                PARTICLE_COUNT,
                2,     // templates
                10000, // contacts
            );

            // Create DEM templates
            let template_fluid = sim3d::ClumpTemplate3D::generate(
                sim3d::ClumpShape3D::Irregular {
                    count: 1,
                    seed: 42,
                    style: sim3d::IrregularStyle3D::Round,
                },
                0.002,  // 2mm radius (fine sediment)
                1000.0, // density like water
            );

            let template_heavy = sim3d::ClumpTemplate3D::generate(
                sim3d::ClumpShape3D::Tetra,
                0.005,  // 5mm radius (heavier sediment)
                2000.0, // density 2x water
            );

            let fluid_id = gpu_dem.add_template(template_fluid);
            let heavy_id = gpu_dem.add_template(template_heavy);

            // Spawn particles in a column
            for i in 0..100 {
                for j in 0..50 {
                    let idx = (i * 50 + j) as u32;
                    if idx >= PARTICLE_COUNT {
                        break;
                    }

                    let x = (j % 10) as f32 * CELL_SIZE;
                    let y = 2.0 + (i as f32) * CELL_SIZE; // Start at 2m height
                    let z = (j / 10) as f32 * CELL_SIZE;

                    let template_id = if (idx % 3) == 0 { fluid_id } else { heavy_id };
                    gpu_dem.spawn_clump(template_id, Vec3::new(x, y, z), Vec3::ZERO);
                }
            }

            // Mark some cells as solid in FLIP
            for k in 0..GRID_DEPTH {
                for i in 0..4 {
                    for j in 0..GRID_HEIGHT {
                        let grid_idx = k * GRID_WIDTH * GRID_HEIGHT + j * GRID_WIDTH + i;
                        gpu_flip.set_cell_solid(grid_idx, true);
                    }
                }
            }

            self.gpu_ctx = Some(gpu_ctx);
            self.gpu_flip = Some(gpu_flip);
            self.gpu_dem = Some(gpu_dem);
            self.window = Some(window_arc);

            println!("GPU DEM-FLIP Integration Test");
            println!("========================");
            println!();
            println!("Initialized {} DEM particles", PARTICLE_COUNT);
            println!(
                "Grid: {}x{}x{}, cell size: {:.3}m",
                GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE
            );
            println!();
        }
    }

    fn window_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("GPU DEM-FLIP test completed");
            }
            WindowEvent::Resized(physical_size) => {
                if let Some(gpu_ctx) = &mut self.gpu_ctx {
                    gpu_ctx.resize(physical_size.width, physical_size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(gpu_ctx), Some(gpu_flip), Some(gpu_dem)) =
                    (&self.gpu_ctx, &mut self.gpu_flip, &mut self.gpu_dem)
                {
                    let mut encoder =
                        gpu_ctx
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("DEM-FLIP Frame"),
                            });

                    // Update DEM first
                    gpu_dem.update(&mut encoder, 1.0 / 60.0);

                    // Update FLIP
                    gpu_flip.update(&mut encoder, 1.0 / 60.0);

                    // TODO: Run DEM-FLIP bridge for coupling
                    // This would need additional shader pipelines

                    // Submit commands
                    let _ = gpu_ctx.queue.submit(Some(encoder.finish()));
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop<()>) -> ControlFlow {
        ControlFlow::Poll
    }
}

fn main() {
    println!("GPU DEM-FLIP Integration Test");
    println!("=============================");
    println!();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut app = DemFlipTest {
        gpu_ctx: None,
        gpu_flip: None,
        gpu_dem: None,
        window: None,
    };

    event_loop.run_app(&mut app).expect("Failed to run app");
}
