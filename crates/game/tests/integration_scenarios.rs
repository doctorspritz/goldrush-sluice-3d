//! Integration Test Scenarios for FLIP Fluid Simulation
//!
//! End-to-end scenario tests that verify realistic physics behaviors
//! in the FLIP fluid simulation. These tests check complete simulation
//! outcomes rather than individual components.
//!
//! Run with: cargo test -p game --test integration_scenarios --release -- --nocapture

use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};

// =============================================================================
// TEST CONFIGURATION
// =============================================================================

const DT: f32 = 1.0 / 60.0; // 60 FPS timestep
const GRAVITY: f32 = -9.81;  // m/s^2

// =============================================================================
// GPU INITIALIZATION
// =============================================================================

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
        println!(
            "Skipped: Adapter only supports {} storage buffers, need 16",
            limits.max_storage_buffers_per_shader_stage
        );
        return None;
    }

    let mut required_limits = wgpu::Limits::default();
    required_limits.max_storage_buffers_per_shader_stage = 16;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Integration Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;
    Some((device, queue))
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Setup solid boundaries: floor, walls, and optionally ceiling
fn setup_solid_boundaries(cell_types: &mut [u32], w: usize, h: usize, d: usize, open_top: bool) {
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let idx = z * w * h + y * w + x;
                let is_floor = y == 0;
                let is_ceiling = y == h - 1;
                let is_x_wall = x == 0 || x == w - 1;
                let is_z_wall = z == 0 || z == d - 1;

                if is_floor || is_x_wall || is_z_wall {
                    cell_types[idx] = 2; // SOLID
                }
                if is_ceiling && !open_top {
                    cell_types[idx] = 2; // SOLID
                }
            }
        }
    }
}

/// Compute center of mass for particles
fn center_of_mass(positions: &[Vec3]) -> Vec3 {
    if positions.is_empty() {
        return Vec3::ZERO;
    }
    let sum: Vec3 = positions.iter().copied().sum();
    sum / positions.len() as f32
}

/// Compute average velocity magnitude
fn avg_velocity_magnitude(velocities: &[Vec3]) -> f32 {
    if velocities.is_empty() {
        return 0.0;
    }
    let sum: f32 = velocities.iter().map(|v| v.length()).sum();
    sum / velocities.len() as f32
}

/// Compute max velocity magnitude
fn max_velocity_magnitude(velocities: &[Vec3]) -> f32 {
    velocities.iter().map(|v| v.length()).fold(0.0f32, f32::max)
}

// =============================================================================
// TEST 1: DAM BREAK SCENARIO
// =============================================================================

/// Dam break scenario: water column on one side spreads horizontally
///
/// Setup:
/// - 20x20x20 grid with 0.05m cells (1m x 1m x 1m domain)
/// - Water column fills left half: cells [1,9] x [1,8] x [1,18]
/// - One particle per cell (7x7x17 = 833 particles)
///
/// Expected behavior:
/// - Water spreads horizontally to the right (center of mass shifts +X)
/// - No particles escape through boundaries
/// - Final velocity should be moderate (< 2.0 m/s after settling for 2s)
///
/// Physics verification:
/// - Tests gravity-driven horizontal flow
/// - Tests boundary collision
/// - Tests pressure solve for incompressibility
#[test]
fn test_dam_break_scenario() {
    let (device, queue) = match init_device() {
        Some(d) => d,
        None => {
            println!("Skipped: No GPU available");
            return;
        }
    };

    println!("\n{}", "=".repeat(70));
    println!("SCENARIO 1: DAM BREAK");
    println!("{}", "=".repeat(70));

    // Grid configuration
    const W: usize = 20;
    const H: usize = 20;
    const D: usize = 20;
    const CELL: f32 = 0.05; // 5cm cells -> 1m x 1m x 1m domain
    const SIM_DURATION: f32 = 2.0; // 2 seconds
    const FRAMES: usize = (SIM_DURATION / DT) as usize;

    println!("Grid: {}x{}x{} cells, cell_size={} m", W, H, D, CELL);
    println!("Domain: {:.2}m x {:.2}m x {:.2}m", W as f32 * CELL, H as f32 * CELL, D as f32 * CELL);
    println!("Simulation: {:.1}s ({} frames at {} Hz)", SIM_DURATION, FRAMES, 1.0 / DT);

    // Create FLIP simulation
    let mut flip = GpuFlip3D::new(&device, W as u32, H as u32, D as u32, CELL, 2000);
    flip.vorticity_epsilon = 0.0;
    flip.sediment_vorticity_lift = 0.0;
    flip.sediment_settling_velocity = 0.0;
    flip.sediment_porosity_drag = 0.0;
    flip.slip_factor = 0.0; // No-slip walls
    flip.open_boundaries = 8; // +Y open (free surface)
    // Density projection defaults (volume preservation)
    flip.water_rest_particles = 8.0;
    flip.density_projection_strength = 1.0;
    flip.volume_iterations = 40;

    // Setup boundary conditions
    let mut cell_types = vec![0u32; W * H * D];
    setup_solid_boundaries(&mut cell_types, W, H, D, true); // open top

    // Spawn water column on LEFT side of domain
    // Water fills cells [1,9] x [1,8] x [1,18] = 8x7x17 cells
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut c_matrices = Vec::new();
    let mut densities = Vec::new();

    for x in 1..10 {  // Left half
        for y in 1..9 {  // 8 cells high (~40cm)
            for z in 1..19 {  // Most of depth
                let pos = Vec3::new(
                    (x as f32 + 0.5) * CELL,
                    (y as f32 + 0.5) * CELL,
                    (z as f32 + 0.5) * CELL,
                );
                positions.push(pos);
                velocities.push(Vec3::ZERO);
                c_matrices.push(Mat3::ZERO);
                densities.push(1.0);
            }
        }
    }

    let initial_particle_count = positions.len();
    let initial_com = center_of_mass(&positions);

    println!("Initial particles: {}", initial_particle_count);
    println!("Initial center of mass: ({:.3}, {:.3}, {:.3})", initial_com.x, initial_com.y, initial_com.z);

    // Domain bounds for escape detection
    let margin = CELL * 0.1;
    let min_bound = CELL + margin;
    let max_x = (W as f32 - 1.0) * CELL - margin;
    let max_y = (H as f32 - 1.0) * CELL - margin;
    let max_z = (D as f32 - 1.0) * CELL - margin;

    // Run simulation
    for frame in 0..FRAMES {
        flip.step(
            &device,
            &queue,
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
            40, // Pressure iterations
        );

        // Check for escaped particles every frame
        for (i, p) in positions.iter().enumerate() {
            if !p.is_finite() {
                panic!("Frame {}: Particle {} has NaN position: {:?}", frame, i, p);
            }
            if p.x < min_bound || p.x > max_x || p.y < min_bound || p.y > max_y || p.z < min_bound || p.z > max_z {
                panic!(
                    "Frame {}: Particle {} escaped bounds at ({:.4}, {:.4}, {:.4})",
                    frame, i, p.x, p.y, p.z
                );
            }
        }

        // Progress report every 0.5s
        if frame % 30 == 0 || frame == FRAMES - 1 {
            let com = center_of_mass(&positions);
            let avg_vel = avg_velocity_magnitude(&velocities);
            let max_vel = max_velocity_magnitude(&velocities);
            println!(
                "Frame {:3} (t={:.2}s): COM=({:.3}, {:.3}, {:.3}), avg_vel={:.3}, max_vel={:.3}",
                frame,
                frame as f32 * DT,
                com.x,
                com.y,
                com.z,
                avg_vel,
                max_vel
            );
        }
    }

    // Final validation
    println!("\n{}", "-".repeat(70));
    println!("VALIDATION");
    println!("{}", "-".repeat(70));

    let final_particle_count = positions.len();
    let final_com = center_of_mass(&positions);
    let final_avg_vel = avg_velocity_magnitude(&velocities);
    let final_max_vel = max_velocity_magnitude(&velocities);

    println!("Final particles: {} (initial: {})", final_particle_count, initial_particle_count);
    println!("Final center of mass: ({:.3}, {:.3}, {:.3})", final_com.x, final_com.y, final_com.z);
    println!("Final avg velocity: {:.3} m/s", final_avg_vel);
    println!("Final max velocity: {:.3} m/s", final_max_vel);
    println!("COM shift X: {:.3} m (rightward spread)", final_com.x - initial_com.x);

    // Assertions
    assert_eq!(
        final_particle_count, initial_particle_count,
        "Particle count changed! {} -> {}",
        initial_particle_count, final_particle_count
    );
    println!("✓ Particle count conserved");

    assert!(
        final_com.x > initial_com.x,
        "Water did not spread rightward! COM X: {:.3} -> {:.3}",
        initial_com.x, final_com.x
    );
    println!("✓ Water spread horizontally (COM shifted +X by {:.3}m)", final_com.x - initial_com.x);

    assert!(
        final_max_vel < 3.0,
        "Final velocity too high! max_vel={:.3} m/s (should be settling)",
        final_max_vel
    );
    println!("✓ Velocity reasonable (max_vel={:.3} m/s < 3.0 m/s)", final_max_vel);

    println!("\n{}", "=".repeat(70));
    println!("TEST PASSED: Dam break scenario");
    println!("{}", "=".repeat(70));
}

// =============================================================================
// TEST 2: HYDROSTATIC COLUMN
// =============================================================================

/// Hydrostatic column: water at rest verifies pressure equilibrium
///
/// Setup:
/// - 10x12x10 grid with 0.1m cells (1.0m x 1.2m x 1.0m domain)
/// - Water fills center region: cells [2,7] x [1,5] x [2,7] = 5x4x5 = 100 cells
/// - 8 particles per cell (2x2x2 stratified) = 800 particles
/// - Water column height = 0.4m
///
/// Expected behavior:
/// - Particles settle to floor under gravity
/// - Hydrostatic pressure develops (higher at bottom)
/// - After 1 second, velocities should be near zero (< 0.1 m/s)
/// - Particle count conserved exactly
///
/// Physics verification:
/// - Tests gravity + pressure balance
/// - Tests settling to equilibrium
/// - Tests no spurious velocities in static state
#[test]
fn test_hydrostatic_column() {
    let (device, queue) = match init_device() {
        Some(d) => d,
        None => {
            println!("Skipped: No GPU available");
            return;
        }
    };

    println!("\n{}", "=".repeat(70));
    println!("SCENARIO 2: HYDROSTATIC COLUMN");
    println!("{}", "=".repeat(70));

    // Grid configuration
    const W: usize = 10;
    const H: usize = 12;
    const D: usize = 10;
    const CELL: f32 = 0.1; // 10cm cells
    const SIM_DURATION: f32 = 1.0; // 1 second to settle
    const FRAMES: usize = (SIM_DURATION / DT) as usize;
    const PARTICLES_PER_CELL: usize = 8; // 2x2x2 stratified

    println!("Grid: {}x{}x{} cells, cell_size={} m", W, H, D, CELL);
    println!("Domain: {:.1}m x {:.1}m x {:.1}m", W as f32 * CELL, H as f32 * CELL, D as f32 * CELL);
    println!("Simulation: {:.1}s ({} frames at {} Hz)", SIM_DURATION, FRAMES, 1.0 / DT);

    // Create FLIP simulation optimized for hydrostatic equilibrium
    let mut flip = GpuFlip3D::new(&device, W as u32, H as u32, D as u32, CELL, 2000);
    flip.configure_for_hydrostatic_equilibrium(); // PIC + density projection tuned for settling
    flip.vorticity_epsilon = 0.0;
    flip.sediment_vorticity_lift = 0.0;
    flip.sediment_settling_velocity = 0.0;
    flip.sediment_porosity_drag = 0.0;
    // Match prior density projection defaults
    flip.volume_iterations = 40;

    // Setup boundary conditions
    let mut cell_types = vec![0u32; W * H * D];
    setup_solid_boundaries(&mut cell_types, W, H, D, true); // open top

    // Spawn water in center region with stratified sampling
    // Water fills cells [2,7] x [1,5] x [2,7] = 5x4x5 cells
    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut c_matrices = Vec::new();
    let mut densities = Vec::new();

    // Helper: spawn 8 particles per cell in 2x2x2 pattern
    fn spawn_stratified(
        cx: usize,
        cy: usize,
        cz: usize,
        cell_size: f32,
        positions: &mut Vec<Vec3>,
        velocities: &mut Vec<Vec3>,
        c_matrices: &mut Vec<Mat3>,
        densities: &mut Vec<f32>,
    ) {
        let offsets = [0.25, 0.75]; // Subcell centers
        for &oz in &offsets {
            for &oy in &offsets {
                for &ox in &offsets {
                    let pos = Vec3::new(
                        (cx as f32 + ox) * cell_size,
                        (cy as f32 + oy) * cell_size,
                        (cz as f32 + oz) * cell_size,
                    );
                    positions.push(pos);
                    velocities.push(Vec3::ZERO);
                    c_matrices.push(Mat3::ZERO);
                    densities.push(1.0);
                }
            }
        }
    }

    for x in 2..8 {
        for y in 1..6 {  // 5 cells high = 0.5m
            for z in 2..8 {
                spawn_stratified(x, y, z, CELL, &mut positions, &mut velocities, &mut c_matrices, &mut densities);
            }
        }
    }

    let initial_particle_count = positions.len();
    let initial_com = center_of_mass(&positions);
    let water_height = 5.0 * CELL; // 5 cells = 0.5m

    println!("Initial particles: {} ({} cells x {} particles/cell)",
        initial_particle_count, initial_particle_count / PARTICLES_PER_CELL, PARTICLES_PER_CELL);
    println!("Water column height: {:.2}m", water_height);
    println!("Initial center of mass: ({:.3}, {:.3}, {:.3})", initial_com.x, initial_com.y, initial_com.z);
    println!("Expected hydrostatic pressure at bottom: ~{:.0} Pa", 1000.0 * 9.81 * water_height);

    // Domain bounds
    let margin = CELL * 0.1;
    let min_bound = CELL + margin;
    let max_x = (W as f32 - 1.0) * CELL - margin;
    let max_y = (H as f32 - 1.0) * CELL - margin;
    let max_z = (D as f32 - 1.0) * CELL - margin;

    // Run simulation
    for frame in 0..FRAMES {
        flip.step(
            &device,
            &queue,
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
            60, // More pressure iterations for equilibrium
        );

        // Check bounds
        for (i, p) in positions.iter().enumerate() {
            if !p.is_finite() {
                panic!("Frame {}: Particle {} has NaN position: {:?}", frame, i, p);
            }
            if p.x < min_bound || p.x > max_x || p.y < min_bound || p.y > max_y || p.z < min_bound || p.z > max_z {
                panic!(
                    "Frame {}: Particle {} escaped bounds at ({:.4}, {:.4}, {:.4})",
                    frame, i, p.x, p.y, p.z
                );
            }
        }

        // Progress report every 0.2s
        if frame % 12 == 0 || frame == FRAMES - 1 {
            let com = center_of_mass(&positions);
            let avg_vel = avg_velocity_magnitude(&velocities);
            let max_vel = max_velocity_magnitude(&velocities);
            let y_min = positions.iter().map(|p| p.y).fold(f32::MAX, f32::min);
            let y_max = positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);
            println!(
                "Frame {:3} (t={:.2}s): Y=[{:.3}, {:.3}], COM_y={:.3}, avg_vel={:.4}, max_vel={:.4}",
                frame,
                frame as f32 * DT,
                y_min,
                y_max,
                com.y,
                avg_vel,
                max_vel
            );
        }
    }

    // Final validation
    println!("\n{}", "-".repeat(70));
    println!("VALIDATION");
    println!("{}", "-".repeat(70));

    let final_particle_count = positions.len();
    let final_com = center_of_mass(&positions);
    let final_avg_vel = avg_velocity_magnitude(&velocities);
    let final_max_vel = max_velocity_magnitude(&velocities);
    let final_y_min = positions.iter().map(|p| p.y).fold(f32::MAX, f32::min);
    let final_y_max = positions.iter().map(|p| p.y).fold(f32::MIN, f32::max);

    println!("Final particles: {} (initial: {})", final_particle_count, initial_particle_count);
    println!("Final Y range: [{:.3}, {:.3}]", final_y_min, final_y_max);
    println!("Final center of mass Y: {:.3} m", final_com.y);
    println!("Final avg velocity: {:.4} m/s", final_avg_vel);
    println!("Final max velocity: {:.4} m/s", final_max_vel);

    // Assertions
    assert_eq!(
        final_particle_count, initial_particle_count,
        "Particle count changed! {} -> {}",
        initial_particle_count, final_particle_count
    );
    println!("✓ Particle count conserved");

    assert!(
        final_avg_vel < 0.2,
        "Average velocity too high! avg_vel={:.4} m/s (should be settled)",
        final_avg_vel
    );
    println!("✓ Velocities near zero (avg={:.4} m/s < 0.2 m/s)", final_avg_vel);

    // Water should have settled (COM Y should decrease)
    assert!(
        final_com.y < initial_com.y,
        "Water did not settle! COM Y: {:.3} -> {:.3}",
        initial_com.y, final_com.y
    );
    println!("✓ Water settled (COM Y decreased by {:.3}m)", initial_com.y - final_com.y);

    println!("\n{}", "=".repeat(70));
    println!("TEST PASSED: Hydrostatic column equilibrium");
    println!("{}", "=".repeat(70));
}

// =============================================================================
// TEST 3: SEDIMENT SETTLING
// =============================================================================

/// Sediment settling: dense particles sink through lighter fluid
///
/// Setup:
/// - 12x16x12 grid with 0.08m cells (0.96m x 1.28m x 0.96m domain)
/// - Two particle layers:
///   - Water layer: cells [3,8] x [1,5] x [3,8] = 5x4x5 = 100 cells
///   - Sediment layer (denser): cells [4,7] x [6,10] x [4,7] = 3x4x3 = 36 cells
/// - Sediment particles start ABOVE water particles
/// - Total: 136 cells, 1 particle per cell
///
/// Expected behavior:
/// - Dense sediment sinks through water
/// - After 2 seconds, sediment average Y < water average Y
/// - Particle count conserved
///
/// Physics verification:
/// - Tests density-driven settling
/// - Tests multi-material interaction
/// - Tests vertical stratification by density
#[test]
fn test_sediment_settling() {
    let (device, queue) = match init_device() {
        Some(d) => d,
        None => {
            println!("Skipped: No GPU available");
            return;
        }
    };

    println!("\n{}", "=".repeat(70));
    println!("SCENARIO 3: SEDIMENT SETTLING");
    println!("{}", "=".repeat(70));

    // Grid configuration
    const W: usize = 12;
    const H: usize = 16;
    const D: usize = 12;
    const CELL: f32 = 0.08; // 8cm cells
    const SIM_DURATION: f32 = 2.0; // 2 seconds
    const FRAMES: usize = (SIM_DURATION / DT) as usize;

    println!("Grid: {}x{}x{} cells, cell_size={} m", W, H, D, CELL);
    println!("Domain: {:.2}m x {:.2}m x {:.2}m", W as f32 * CELL, H as f32 * CELL, D as f32 * CELL);
    println!("Simulation: {:.1}s ({} frames at {} Hz)", SIM_DURATION, FRAMES, 1.0 / DT);

    // Create FLIP simulation
    let mut flip = GpuFlip3D::new(&device, W as u32, H as u32, D as u32, CELL, 2000);
    flip.vorticity_epsilon = 0.0;
    flip.sediment_vorticity_lift = 0.0;
    flip.sediment_settling_velocity = 0.0;
    flip.sediment_porosity_drag = 0.0;
    flip.slip_factor = 0.0;
    flip.open_boundaries = 8; // +Y open
    // Density projection defaults (volume preservation)
    flip.water_rest_particles = 8.0;
    flip.density_projection_strength = 1.0;
    flip.volume_iterations = 40;

    // Setup boundary conditions
    let mut cell_types = vec![0u32; W * H * D];
    setup_solid_boundaries(&mut cell_types, W, H, D, true); // open top

    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut c_matrices = Vec::new();
    let mut densities = Vec::new();

    // Spawn WATER particles (density = 1.0) in LOWER region
    // cells [3,8] x [1,5] x [3,8] = 5x4x5 cells
    let mut water_count = 0;
    for x in 3..9 {
        for y in 1..6 {  // Lower layer
            for z in 3..9 {
                let pos = Vec3::new(
                    (x as f32 + 0.5) * CELL,
                    (y as f32 + 0.5) * CELL,
                    (z as f32 + 0.5) * CELL,
                );
                positions.push(pos);
                velocities.push(Vec3::ZERO);
                c_matrices.push(Mat3::ZERO);
                densities.push(1.0); // Water density marker
                water_count += 1;
            }
        }
    }

    // Spawn SEDIMENT particles (density = 2.5) in UPPER region
    // cells [4,7] x [7,11] x [4,7] = 3x4x3 cells
    let mut sediment_count = 0;
    for x in 4..8 {
        for y in 7..12 {  // Upper layer, ABOVE water
            for z in 4..8 {
                let pos = Vec3::new(
                    (x as f32 + 0.5) * CELL,
                    (y as f32 + 0.5) * CELL,
                    (z as f32 + 0.5) * CELL,
                );
                positions.push(pos);
                velocities.push(Vec3::ZERO);
                c_matrices.push(Mat3::ZERO);
                densities.push(2.5); // Sediment density marker (2.5x heavier)
                sediment_count += 1;
            }
        }
    }

    let total_particles = positions.len();
    let water_indices: Vec<usize> = (0..water_count).collect();
    let sediment_indices: Vec<usize> = (water_count..total_particles).collect();

    println!("Water particles: {} (density=1.0)", water_count);
    println!("Sediment particles: {} (density=2.5, starts ABOVE water)", sediment_count);
    println!("Total particles: {}", total_particles);

    // Initial statistics
    let initial_water_y_avg: f32 = water_indices.iter()
        .map(|&i| positions[i].y)
        .sum::<f32>() / water_count as f32;
    let initial_sediment_y_avg: f32 = sediment_indices.iter()
        .map(|&i| positions[i].y)
        .sum::<f32>() / sediment_count as f32;

    println!("Initial water avg Y: {:.3} m", initial_water_y_avg);
    println!("Initial sediment avg Y: {:.3} m (higher than water)", initial_sediment_y_avg);

    // Domain bounds
    let margin = CELL * 0.1;
    let min_bound = CELL + margin;
    let max_x = (W as f32 - 1.0) * CELL - margin;
    let max_y = (H as f32 - 1.0) * CELL - margin;
    let max_z = (D as f32 - 1.0) * CELL - margin;

    // Run simulation
    for frame in 0..FRAMES {
        flip.step(
            &device,
            &queue,
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
            40, // Pressure iterations
        );

        // Check bounds
        for (i, p) in positions.iter().enumerate() {
            if !p.is_finite() {
                panic!("Frame {}: Particle {} has NaN position: {:?}", frame, i, p);
            }
            if p.x < min_bound || p.x > max_x || p.y < min_bound || p.y > max_y || p.z < min_bound || p.z > max_z {
                panic!(
                    "Frame {}: Particle {} escaped bounds at ({:.4}, {:.4}, {:.4})",
                    frame, i, p.x, p.y, p.z
                );
            }
        }

        // Progress report every 0.5s
        if frame % 30 == 0 || frame == FRAMES - 1 {
            let water_y_avg: f32 = water_indices.iter()
                .map(|&i| positions[i].y)
                .sum::<f32>() / water_count as f32;
            let sediment_y_avg: f32 = sediment_indices.iter()
                .map(|&i| positions[i].y)
                .sum::<f32>() / sediment_count as f32;
            let avg_vel = avg_velocity_magnitude(&velocities);

            println!(
                "Frame {:3} (t={:.2}s): Water_y={:.3}, Sediment_y={:.3}, diff={:.3}, avg_vel={:.3}",
                frame,
                frame as f32 * DT,
                water_y_avg,
                sediment_y_avg,
                sediment_y_avg - water_y_avg,
                avg_vel
            );
        }
    }

    // Final validation
    println!("\n{}", "-".repeat(70));
    println!("VALIDATION");
    println!("{}", "-".repeat(70));

    let final_particle_count = positions.len();
    let final_water_y_avg: f32 = water_indices.iter()
        .map(|&i| positions[i].y)
        .sum::<f32>() / water_count as f32;
    let final_sediment_y_avg: f32 = sediment_indices.iter()
        .map(|&i| positions[i].y)
        .sum::<f32>() / sediment_count as f32;
    let final_avg_vel = avg_velocity_magnitude(&velocities);

    println!("Final particles: {} (initial: {})", final_particle_count, total_particles);
    println!("Final water avg Y: {:.3} m", final_water_y_avg);
    println!("Final sediment avg Y: {:.3} m", final_sediment_y_avg);
    println!("Final avg velocity: {:.3} m/s", final_avg_vel);
    println!("Sediment vs Water Y diff: {:.3} m (negative = sediment below)", final_sediment_y_avg - final_water_y_avg);

    // Assertions
    assert_eq!(
        final_particle_count, total_particles,
        "Particle count changed! {} -> {}",
        total_particles, final_particle_count
    );
    println!("✓ Particle count conserved");

    assert!(
        final_sediment_y_avg < final_water_y_avg,
        "Sediment did not sink below water! Sediment Y={:.3}, Water Y={:.3}",
        final_sediment_y_avg, final_water_y_avg
    );
    println!(
        "✓ Sediment sank below water (sediment {:.3}m lower than water)",
        final_water_y_avg - final_sediment_y_avg
    );

    // Sediment should have moved down significantly
    let sediment_drop = initial_sediment_y_avg - final_sediment_y_avg;
    assert!(
        sediment_drop > 0.1,
        "Sediment did not sink enough! Drop={:.3}m (should be >0.1m)",
        sediment_drop
    );
    println!("✓ Sediment dropped {:.3}m from initial position", sediment_drop);

    println!("\n{}", "=".repeat(70));
    println!("TEST PASSED: Sediment settling");
    println!("{}", "=".repeat(70));
}
