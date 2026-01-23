//! Property-based tests for FLIP fluid simulation using proptest
//!
//! These tests verify physics invariants hold across random initial conditions:
//! - No NaN values in positions/velocities
//! - Particle count conservation
//! - Velocity magnitude bounds
//! - Spatial bounds containment

use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};
use proptest::prelude::*;

// Simulation domain constants
const GRID_WIDTH: u32 = 16;
const GRID_HEIGHT: u32 = 16;
const GRID_DEPTH: u32 = 16;
const CELL_SIZE: f32 = 0.1;
const MAX_PARTICLES: usize = 100;
const DT: f32 = 1.0 / 60.0;

// Physics bounds
const MAX_EXPECTED_VELOCITY: f32 = 50.0; // m/s (terminal velocity + margin)
const GRAVITY: f32 = -9.81;

// Test configuration
const SIMULATION_STEPS: usize = 10;

/// Initialize GPU device for testing
fn init_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    let limits = adapter.limits();
    if limits.max_storage_buffers_per_shader_stage < 16 {
        return None;
    }

    let mut required_limits = wgpu::Limits::default();
    required_limits.max_storage_buffers_per_shader_stage = 16;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Proptest Device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;
    Some((device, queue))
}

/// Setup solid boundaries (floor and walls, open top)
fn setup_boundaries(cell_types: &mut [u32], w: usize, h: usize, d: usize) {
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let idx = z * w * h + y * w + x;
                let is_floor = y == 0;
                let is_x_wall = x == 0 || x == w - 1;
                let is_z_wall = z == 0 || z == d - 1;

                if is_floor || is_x_wall || is_z_wall {
                    cell_types[idx] = 2; // SOLID
                }
            }
        }
    }
}

/// Strategy to generate valid particle positions within the domain
fn valid_position() -> impl Strategy<Value = Vec3> {
    // Keep particles away from boundaries (1.5 cells margin)
    let margin = CELL_SIZE * 1.5;
    let max_x = GRID_WIDTH as f32 * CELL_SIZE - margin;
    let max_y = GRID_HEIGHT as f32 * CELL_SIZE - margin;
    let max_z = GRID_DEPTH as f32 * CELL_SIZE - margin;

    (
        margin..max_x,
        margin..max_y,
        margin..max_z,
    ).prop_map(|(x, y, z)| Vec3::new(x, y, z))
}

/// Strategy to generate reasonable initial velocities
fn valid_velocity() -> impl Strategy<Value = Vec3> {
    // Initial velocities between -5.0 and 5.0 m/s per axis
    (
        -5.0f32..5.0f32,
        -5.0f32..5.0f32,
        -5.0f32..5.0f32,
    ).prop_map(|(x, y, z)| Vec3::new(x, y, z))
}

/// Strategy to generate a set of particles
fn particle_set() -> impl Strategy<Value = (Vec<Vec3>, Vec<Vec3>)> {
    // Generate 10-100 particles
    (10usize..=100)
        .prop_flat_map(|count| {
            (
                prop::collection::vec(valid_position(), count..=count),
                prop::collection::vec(valid_velocity(), count..=count),
            )
        })
}

/// Run a short simulation and return final state
fn run_simulation(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut positions: Vec<Vec3>,
    mut velocities: Vec<Vec3>,
    steps: usize,
) -> (Vec<Vec3>, Vec<Vec3>) {
    let w = GRID_WIDTH as usize;
    let h = GRID_HEIGHT as usize;
    let d = GRID_DEPTH as usize;

    // Setup simulation
    let mut flip = GpuFlip3D::new(device, GRID_WIDTH, GRID_HEIGHT, GRID_DEPTH, CELL_SIZE, MAX_PARTICLES);
    flip.vorticity_epsilon = 0.0;
    flip.sediment_vorticity_lift = 0.0;
    flip.sediment_settling_velocity = 0.0;
    flip.sediment_porosity_drag = 0.0;
    flip.slip_factor = 0.0;
    flip.open_boundaries = 8; // +Y open for free surface

    let mut c_matrices = vec![Mat3::ZERO; positions.len()];
    let densities = vec![1.0f32; positions.len()];
    let mut cell_types = vec![0u32; w * h * d];
    setup_boundaries(&mut cell_types, w, h, d);

    // Run simulation
    for _ in 0..steps {
        flip.step(
            device,
            queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            DT,
            GRAVITY,
            0.0,
            40, // Moderate pressure iterations for proptest
        );
    }

    (positions, velocities)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// Property: Particle positions and velocities must never contain NaN
    #[test]
    fn test_particle_positions_no_nan((positions, velocities) in particle_set()) {
        let (device, queue) = match init_device() {
            Some(d) => d,
            None => {
                // Skip test if no GPU available
                return Ok(());
            }
        };

        let (final_positions, final_velocities) = run_simulation(
            &device,
            &queue,
            positions.clone(),
            velocities.clone(),
            SIMULATION_STEPS,
        );

        // Check for NaN in positions
        for (i, pos) in final_positions.iter().enumerate() {
            prop_assert!(
                pos.is_finite(),
                "Position {} contains NaN/Inf: {:?}", i, pos
            );
        }

        // Check for NaN in velocities
        for (i, vel) in final_velocities.iter().enumerate() {
            prop_assert!(
                vel.is_finite(),
                "Velocity {} contains NaN/Inf: {:?}", i, vel
            );
        }
    }

    /// Property: Particle count must be conserved
    #[test]
    fn test_particle_count_conserved((positions, velocities) in particle_set()) {
        let (device, queue) = match init_device() {
            Some(d) => d,
            None => return Ok(()),
        };

        let initial_count = positions.len();

        let (final_positions, _) = run_simulation(
            &device,
            &queue,
            positions,
            velocities,
            SIMULATION_STEPS,
        );

        prop_assert_eq!(
            final_positions.len(),
            initial_count,
            "Particle count changed during simulation"
        );
    }

    /// Property: Velocities must stay within reasonable physical bounds
    #[test]
    fn test_velocities_bounded((positions, velocities) in particle_set()) {
        let (device, queue) = match init_device() {
            Some(d) => d,
            None => return Ok(()),
        };

        let (_, final_velocities) = run_simulation(
            &device,
            &queue,
            positions,
            velocities,
            SIMULATION_STEPS,
        );

        for (i, vel) in final_velocities.iter().enumerate() {
            let speed = vel.length();
            prop_assert!(
                speed <= MAX_EXPECTED_VELOCITY,
                "Particle {} has excessive velocity: {} m/s (max: {} m/s)\nVelocity: {:?}",
                i, speed, MAX_EXPECTED_VELOCITY, vel
            );
        }
    }

    /// Property: Particles must stay within the simulation domain
    #[test]
    fn test_positions_in_bounds((positions, velocities) in particle_set()) {
        let (device, queue) = match init_device() {
            Some(d) => d,
            None => return Ok(()),
        };

        let (final_positions, _) = run_simulation(
            &device,
            &queue,
            positions,
            velocities,
            SIMULATION_STEPS,
        );

        let max_x = GRID_WIDTH as f32 * CELL_SIZE;
        let max_y = GRID_HEIGHT as f32 * CELL_SIZE;
        let max_z = GRID_DEPTH as f32 * CELL_SIZE;
        let min_bound = 0.0;

        for (i, pos) in final_positions.iter().enumerate() {
            prop_assert!(
                pos.x >= min_bound && pos.x <= max_x,
                "Particle {} X out of bounds: {} (should be in [0, {}])", i, pos.x, max_x
            );
            prop_assert!(
                pos.y >= min_bound && pos.y <= max_y,
                "Particle {} Y out of bounds: {} (should be in [0, {}])", i, pos.y, max_y
            );
            prop_assert!(
                pos.z >= min_bound && pos.z <= max_z,
                "Particle {} Z out of bounds: {} (should be in [0, {}])", i, pos.z, max_z
            );
        }
    }
}

#[cfg(test)]
mod determinism_tests {
    use super::*;

    /// Property: Same initial conditions should produce identical results (determinism)
    #[test]
    fn test_simulation_deterministic() {
        let (device, queue) = match init_device() {
            Some(d) => d,
            None => {
                println!("Skipped: No GPU");
                return;
            }
        };

        // Fixed initial conditions
        let positions = vec![
            Vec3::new(0.5, 0.8, 0.5),
            Vec3::new(0.7, 0.9, 0.7),
            Vec3::new(0.3, 1.0, 0.3),
        ];
        let velocities = vec![
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.1, -0.5, 0.1),
            Vec3::new(-0.1, -0.8, -0.1),
        ];

        // Run simulation twice
        let (positions1, velocities1) = run_simulation(
            &device,
            &queue,
            positions.clone(),
            velocities.clone(),
            5,
        );

        let (positions2, velocities2) = run_simulation(
            &device,
            &queue,
            positions.clone(),
            velocities.clone(),
            5,
        );

        // Results should be identical
        for (i, (p1, p2)) in positions1.iter().zip(positions2.iter()).enumerate() {
            let diff = (*p1 - *p2).length();
            assert!(
                diff < 1e-6,
                "Position {} differs between runs: {:?} vs {:?} (diff: {})",
                i, p1, p2, diff
            );
        }

        for (i, (v1, v2)) in velocities1.iter().zip(velocities2.iter()).enumerate() {
            let diff = (*v1 - *v2).length();
            assert!(
                diff < 1e-6,
                "Velocity {} differs between runs: {:?} vs {:?} (diff: {})",
                i, v1, v2, diff
            );
        }
    }
}
