use game::gpu::flip_3d::GpuFlip3D;
use game::gpu::sph_3d::GpuSph3D;
use glam::{Mat3, Vec3};
use std::sync::Arc;

pub const CELL_FLUID: u32 = 1;
pub const CELL_SOLID: u32 = 2;

pub struct HarnessConfig {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub cell_size: f32,
    pub max_particles: usize,
    pub rest_density: f32,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            width: 20,
            height: 20,
            depth: 20,
            cell_size: 0.05,
            max_particles: 10000,
            rest_density: 8.0,
        }
    }
}

pub struct SimulationSnapshot {
    pub step: usize,
    pub nan_count: usize,
    pub max_velocity: f32,
    pub max_density: f32,
    pub min_density: f32,
    pub mean_density: f32,
    pub max_density_error: f32,
    pub positions_y_min: f32,
}

pub struct TestResult {
    pub passed: bool,
    pub failure_reason: Option<String>,
    pub snapshots: Vec<SimulationSnapshot>,
}

pub enum ScenarioType {
    GravityDrop,
    HydrostaticEquilibrium,
}

pub enum Backend {
    Flip,
    Sph, // Placeholder for now
}

pub trait Simulation {
    fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &mut [glam::Vec3],
        velocities: &mut [glam::Vec3],
        densities: &[f32], // Read-only input for now, sim maintains its own density
        cell_types: &[u32],
        dt: f32,
    );
}

struct FlipAdapter {
    sim: GpuFlip3D,
    c_matrices: Vec<glam::Mat3>,
}

impl FlipAdapter {
    fn new(device: &wgpu::Device, config: &HarnessConfig) -> Self {
        let mut sim = GpuFlip3D::new(
            device,
            config.width,
            config.height,
            config.depth,
            config.cell_size,
            config.max_particles,
        );
        // Disable extra physics features for core testing
        sim.vorticity_epsilon = 0.0;
        sim.sediment_vorticity_lift = 0.0;
        sim.sediment_settling_velocity = 0.0;
        sim.sediment_porosity_drag = 0.0;
        
        Self {
            sim,
            c_matrices: vec![glam::Mat3::ZERO; config.max_particles],
        }
    }
}

impl Simulation for FlipAdapter {
    fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &mut [glam::Vec3],
        velocities: &mut [glam::Vec3],
        densities: &[f32],
        cell_types: &[u32],
        dt: f32,
    ) {
         // Resize c_matrices if needed
        if self.c_matrices.len() < positions.len() {
            self.c_matrices.resize(positions.len(), glam::Mat3::ZERO);
        }

        self.sim.step(
            device,
            queue,
            positions,
            velocities,
            &mut self.c_matrices[..positions.len()],
            densities,
            cell_types,
            None,
            None,
            dt,
            -9.81,
            0.0,
            40, // Pressure iterations
        );
    }
}

struct SphAdapter {
    sim: GpuSph3D,
}

impl SphAdapter {
     fn new(device: &wgpu::Device, config: &HarnessConfig) -> Self {
         let sim = GpuSph3D::new(
             device,
             config.width,
             config.height,
             config.depth,
             config.cell_size,
             config.max_particles,
         );
         Self { sim }
     }
}

impl Simulation for SphAdapter {
    fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &mut [glam::Vec3],
        velocities: &mut [glam::Vec3],
        _densities: &[f32],
        _cell_types: &[u32],
        dt: f32,
    ) {
        self.sim.step(device, queue, positions, velocities, dt);
    }
}


pub fn init_device_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
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
        eprintln!(
            "GPU adapter only supports {} storage buffers (need 16+); skipping test.",
            limits.max_storage_buffers_per_shader_stage
        );
        return None;
    }

    let mut required_limits = wgpu::Limits::default();
    required_limits.max_storage_buffers_per_shader_stage = 16;
    required_limits.max_compute_invocations_per_workgroup = 256;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Headless Harness Device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;

    Some((device, queue))
}

pub fn run_scenario(backend: Backend, scenario_type: ScenarioType, steps: usize, config: HarnessConfig) -> TestResult {
    let (device, queue) = match init_device_queue() {
        Some(h) => h,
        None => return TestResult { passed: true, failure_reason: Some("Skipped: No GPU".to_string()), snapshots: vec![] },
    };

    let mut sim: Box<dyn Simulation> = match backend {
        Backend::Flip => Box::new(FlipAdapter::new(&device, &config)),
        Backend::Sph => Box::new(SphAdapter::new(&device, &config)),
    };

    let mut cell_types = vec![CELL_FLUID; (config.width * config.height * config.depth) as usize];
    // Solid boundary box
    for z in 0..config.depth {
        for y in 0..config.height {
            for x in 0..config.width {
                if x == 0 || x == config.width - 1 || y == 0 || y == config.height - 1 || z == 0 || z == config.depth - 1 {
                    let idx = (z * config.width * config.height + y * config.width + x) as usize;
                    cell_types[idx] = CELL_SOLID;
                }
            }
        }
    }

    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    // c_matrices moved to FlipAdapter

    // Initialize particles based on scenario
    match scenario_type {
        ScenarioType::GravityDrop => {
            // Block of particles suspended in air
            let start_x = 5;
            let start_y = 10;
            let start_z = 5;
            let size = 6;
            for x in 0..size {
                for y in 0..size {
                    for z in 0..size {
                        let pos = Vec3::new(
                            (start_x + x) as f32 * config.cell_size + 0.5 * config.cell_size,
                            (start_y + y) as f32 * config.cell_size + 0.5 * config.cell_size,
                            (start_z + z) as f32 * config.cell_size + 0.5 * config.cell_size,
                        );
                        positions.push(pos);
                        velocities.push(Vec3::ZERO);
                        densities.push(1.0);
                        // c_matrices handled in adapter
                    }
                }
            }
        }
        ScenarioType::HydrostaticEquilibrium => {
             // Block of particles resting on floor
            let start_x = 5;
            let start_y = 1; // On floor (y=1 is first fluid cell)
            let start_z = 5;
            let size = 8;
             for x in 0..size {
                for y in 0..size {
                    for z in 0..size {
                        let pos = Vec3::new(
                            (start_x + x) as f32 * config.cell_size + 0.5 * config.cell_size,
                            (start_y + y) as f32 * config.cell_size + 0.5 * config.cell_size,
                            (start_z + z) as f32 * config.cell_size + 0.5 * config.cell_size,
                        );
                        positions.push(pos);
                        velocities.push(Vec3::ZERO);
                        densities.push(1.0);
                        // c_matrices handled in adapter
                    }
                }
            }
        }
    }

    let dt = 1.0 / 60.0;
    let mut snapshots = Vec::new();

    for step in 0..steps {
        sim.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &densities,
            &cell_types,
            dt,
        );

        // Compute Metrics
        let mut nan_count = 0;
        let mut max_vel = 0.0f32;
        let mut max_rho = 0.0f32;
        let mut min_rho = f32::MAX;
        let mut sum_rho = 0.0f32;
        let mut min_y = f32::MAX;

        // Note: For real metric calculation we need actual density readback which isn't fully exposed in step()
        // step() reads back positions/velocities but GpuFlip3D keeps densities on GPU mostly.
        // We will approximate density error based on particle spacing if needed, 
        // OR we can assume densities vector passed in is updated? No, densities is &densities (read-only for p2g).
        // The simulation doesn't copy densities back to CPU by default!
        // For this MVP, we will only check kinematics (Positions/Velocities).
        // Resolving density readback is a later task.
        
        for (i, _) in positions.iter().enumerate() {
            let p = positions[i];
            let v = velocities[i];
            
            if p.is_nan() || v.is_nan() {
                nan_count += 1;
            }
            let v_mag = v.length();
            if v_mag > max_vel { max_vel = v_mag; }
            if p.y < min_y { min_y = p.y; }
        }

        snapshots.push(SimulationSnapshot {
            step,
            nan_count,
            max_velocity: max_vel,
            max_density: 0.0, // Placeholder
            min_density: 0.0, // Placeholder
            mean_density: 0.0, // Placeholder
            max_density_error: 0.0, // Placeholder
            positions_y_min: min_y,
        });

        // Invariant Checks on the fly
        if nan_count > 0 {
            return TestResult { passed: false, failure_reason: Some(format!("NaN Detected at step {}", step)), snapshots };
        }
        
        match scenario_type {
            ScenarioType::HydrostaticEquilibrium => {
                if max_vel > 1.0 { // loose check
                     return TestResult { passed: false, failure_reason: Some(format!("Velocity Explosion at step {}: {}", step, max_vel)), snapshots };
                }
            }
             ScenarioType::GravityDrop => {
                 if min_y < config.cell_size * 0.5 { // Floor penetration (y < 0.5 is solid wall)
                      return TestResult { passed: false, failure_reason: Some(format!("Floor Penetration at step {}: y={}", step, min_y)), snapshots };
                 }
             }
        }
    }

    TestResult { passed: true, failure_reason: None, snapshots }
}

#[test]
fn test_gravity_drop() {
    let config = HarnessConfig::default();
    let result = run_scenario(Backend::Flip, ScenarioType::GravityDrop, 100, config);
    if !result.passed {
        panic!("Gravity Drop Failed: {:?}", result.failure_reason);
    }
}

#[test]
fn test_hydrostatic_equilibrium() {
    let config = HarnessConfig::default();
    let result = run_scenario(Backend::Flip, ScenarioType::HydrostaticEquilibrium, 50, config);
    if !result.passed {
        panic!("Hydrostatic Equilibrium Failed: {:?}", result.failure_reason);
    }
}

#[test]
fn test_sph_skeleton() {
    let config = HarnessConfig::default();
    // Just run a few steps to ensure it doesn't crash
    let result = run_scenario(Backend::Sph, ScenarioType::GravityDrop, 10, config);
    // It will pass because placeholder does nothing
    if !result.passed {
        panic!("SPH Skeleton Failed: {:?}", result.failure_reason);
    }
}

#[test]
fn test_sph_gravity_drop_fails() {
    let config = HarnessConfig::default();
    let result = run_scenario(Backend::Sph, ScenarioType::GravityDrop, 50, config);
    
    assert!(result.passed, "Empty solver should result in no errors (false positive), passed: {:?}", result.passed);
    
    let snapshot = result.snapshots.last().unwrap();
    // Initial y is (10 * 0.05) + 0.025 = 0.525
    let expected_y = (10.0 * 0.05) + (0.05 * 0.5);
    
    // With empty solver, particles should NOT move.
    if (snapshot.positions_y_min - expected_y).abs() > 0.001 {
         panic!("Particles moved! Solver is not empty? y_min={}, expected={}", snapshot.positions_y_min, expected_y);
    }
}
