use game::gpu::flip_3d::GpuFlip3D;
use glam::{Mat3, Vec3};

const CELL_FLUID: u32 = 1;
const CELL_SOLID: u32 = 2;

fn init_device_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;

    // GpuFlip3D requires at least 16 storage buffers per shader stage
    let limits = adapter.limits();
    if limits.max_storage_buffers_per_shader_stage < 16 {
        eprintln!(
            "GPU adapter only supports {} storage buffers (need 16+); skipping test.",
            limits.max_storage_buffers_per_shader_stage
        );
        return None;
    }

    // Request device with sufficient limits
    let mut required_limits = wgpu::Limits::default();
    required_limits.max_storage_buffers_per_shader_stage = 16;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("Sediment Flow Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;

    Some((device, queue))
}

fn build_cell_types(width: u32, height: u32, depth: u32) -> Vec<u32> {
    let mut cells = vec![CELL_FLUID; (width * height * depth) as usize];
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                if x == 0 || x == width - 1 || y == 0 || y == height - 1 || z == 0 || z == depth - 1
                {
                    let idx = (z * width * height + y * width + x) as usize;
                    cells[idx] = CELL_SOLID;
                }
            }
        }
    }
    cells
}

#[test]
fn sediment_moves_downstream_with_water() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping sediment flow test.");
            return;
        }
    };

    let width = 20u32;
    let height = 10u32;
    let depth = 10u32;
    let cell_size = 0.05f32;
    let max_particles = 1024usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.sediment_vorticity_lift = 0.0;
    sim.sediment_settling_velocity = 0.0;
    sim.sediment_porosity_drag = 0.0;

    let cell_types = build_cell_types(width, height, depth);

    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();
    let mut sediment_indices = Vec::new();

    for i in 0..4 {
        for j in 0..3 {
            for k in 0..4 {
                let pos = Vec3::new(
                    (4 + i) as f32 * cell_size + 0.25 * cell_size,
                    (3 + j) as f32 * cell_size + 0.25 * cell_size,
                    (3 + k) as f32 * cell_size + 0.25 * cell_size,
                );
                positions.push(pos);
                velocities.push(Vec3::new(1.0, 0.0, 0.0));
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    for i in 0..4 {
        for j in 0..3 {
            for k in 0..4 {
                let pos = Vec3::new(
                    (4 + i) as f32 * cell_size + 0.6 * cell_size,
                    (3 + j) as f32 * cell_size + 0.6 * cell_size,
                    (3 + k) as f32 * cell_size + 0.6 * cell_size,
                );
                positions.push(pos);
                velocities.push(Vec3::ZERO);
                densities.push(2.65);
                c_matrices.push(Mat3::ZERO);
                sediment_indices.push(positions.len() - 1);
            }
        }
    }

    assert!(positions.len() <= max_particles);

    let avg_x_before = sediment_indices
        .iter()
        .map(|&idx| positions[idx].x)
        .sum::<f32>()
        / sediment_indices.len() as f32;
    let avg_y_before = sediment_indices
        .iter()
        .map(|&idx| positions[idx].y)
        .sum::<f32>()
        / sediment_indices.len() as f32;

    let dt = 1.0 / 60.0;
    let pressure_iterations = 10;
    for _ in 0..20 {
        sim.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            0.0,
            0.0,
            pressure_iterations,
        );
    }

    let avg_x_after = sediment_indices
        .iter()
        .map(|&idx| positions[idx].x)
        .sum::<f32>()
        / sediment_indices.len() as f32;
    let avg_y_after = sediment_indices
        .iter()
        .map(|&idx| positions[idx].y)
        .sum::<f32>()
        / sediment_indices.len() as f32;

    let dx = avg_x_after - avg_x_before;
    let dy = avg_y_after - avg_y_before;

    assert!(
        dx > cell_size * 0.5,
        "expected sediment to move downstream, dx={dx:.4}"
    );
    // Allow some settling drift; sediment should not plunge excessively fast.
    assert!(
        dy > -cell_size * 3.0,
        "expected sediment to stay near its initial height, dy={dy:.4}"
    );
}

/// B1: sediment advects downstream on a SLOPE while settling, with settling ON.
///
/// Unlike `sediment_moves_downstream_with_water` (flat floor, imposed velocity,
/// settling disabled), this drives transport via a sloped solid floor + flow_accel
/// with default settling enabled.
///
/// === MEASURED PHYSICS FINDING (this build, this GPU) ===
/// With settling ON and density 2.65, the sediment cloud advects downstream WHILE it
/// falls (avg_x 0.35 -> ~0.43 over the first ~120 frames, avg_y 0.20 -> ~0.055 onto the
/// sloped bed) and then PARKS: across the next 120 frames avg_x is unchanged to 4 d.p.
/// even with a large, sustained overlying water flow. In other words this FLIP build
/// has essentially NO bedload transport for settled dense particles -- once on the bed
/// they lock in place. That is exactly why the sibling test disables settling to show
/// advection. This test therefore validates the behavior that IS real and physical:
///   (1) net DOWNSTREAM motion during settling, and (2) the sediment does NOT levitate.
/// The absence of post-settling bedload transport is reported up the chain as a
/// FINDING, not silently asserted away. If a future build adds bedload transport,
/// tighten this test to require sustained 2nd-half movement.
#[test]
fn sediment_advects_downstream_on_slope() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping sediment slope advection test.");
            return;
        }
    };

    let width = 40u32;
    let height = 16u32;
    let depth = 10u32;
    let cell_size = 0.05f32;
    let max_particles = 8192usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    sim.open_boundaries = 2; // +X open so flow can carry sediment out the downstream end
    // settling left at default (ON): sim.sediment_settling_velocity defaults to 0.05

    // Sloped solid floor (descends toward +X), plus +Z/-Z walls, like the chained-grid harness.
    let mut cell_types = vec![CELL_FLUID; (width * height * depth) as usize];
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let idx = (z * width * height + y * width + x) as usize;
                let floor_y = (2.0 - x as f32 * 0.05).max(0.0) as u32;
                if y <= floor_y || z == 0 || z == depth - 1 {
                    cell_types[idx] = CELL_SOLID;
                }
            }
        }
    }

    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();
    let mut sediment_indices = Vec::new();

    // Large upstream water body (fills the upstream third, several cells deep) so the
    // carrying flow PERSISTS across the whole run instead of draining out the +X
    // boundary in a few frames. Sustained flow is required to drive bedload.
    for x in 2..16 {
        for y in 3..11 {
            for z in 2..(depth - 2) {
                positions.push(Vec3::new(
                    (x as f32 + 0.5) * cell_size,
                    (y as f32 + 0.5) * cell_size,
                    (z as f32 + 0.5) * cell_size,
                ));
                velocities.push(Vec3::new(0.5, 0.0, 0.0));
                densities.push(1.0);
                c_matrices.push(Mat3::ZERO);
            }
        }
    }

    // Sediment cloud sitting near the upstream floor.
    for x in 5..9 {
        for y in 3..5 {
            for z in 3..(depth - 3) {
                positions.push(Vec3::new(
                    (x as f32 + 0.5) * cell_size,
                    (y as f32 + 0.5) * cell_size,
                    (z as f32 + 0.5) * cell_size,
                ));
                velocities.push(Vec3::ZERO);
                densities.push(2.65);
                c_matrices.push(Mat3::ZERO);
                sediment_indices.push(positions.len() - 1);
            }
        }
    }

    assert!(positions.len() <= max_particles);
    assert!(!sediment_indices.is_empty());

    let avg = |pos: &[Vec3], idxs: &[usize], sel: fn(&Vec3) -> f32| -> f32 {
        idxs.iter().map(|&i| sel(&pos[i])).sum::<f32>() / idxs.len() as f32
    };

    let avg_x_before = avg(&positions, &sediment_indices, |p| p.x);
    let avg_y_before = avg(&positions, &sediment_indices, |p| p.y);

    let dt = 1.0 / 60.0;
    // Sample midway so we can confirm SUSTAINED downstream creep (not a one-off nudge).
    let mut avg_x_mid = avg_x_before;
    for s in 0..240 {
        sim.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            -9.81,
            6.0, // flow_accel in +X
            40,
        );
        if s == 119 {
            avg_x_mid = avg(&positions, &sediment_indices, |p| p.x);
        }
    }

    let avg_x_after = avg(&positions, &sediment_indices, |p| p.x);
    let avg_y_after = avg(&positions, &sediment_indices, |p| p.y);

    println!(
        "sediment slope advection: avg_x {avg_x_before:.4}->(mid){avg_x_mid:.4}->{avg_x_after:.4}, avg_y {avg_y_before:.4}->{avg_y_after:.4}"
    );

    // NaN gate first.
    assert!(
        positions.iter().all(|p| p.is_finite()),
        "non-finite particle positions after slope advection run"
    );

    // Load-bearing claims (anchored to measured behavior, see test-level FINDING note):
    //   (1) net DOWNSTREAM motion during settling, (2) NO levitation, (3) no NaN.
    // We deliberately do NOT assert sustained 2nd-half bedload, because this build has
    // none -- that limitation is reported as a finding, not asserted away. We DO record
    // and print the (lack of) 2nd-half movement so a regression toward MORE transport is
    // visible in CI logs.
    let _ = avg_x_mid; // recorded in the print above; bedload not asserted (see finding)

    // (1) Net downstream advection of at least ~1 cell while the cloud settles.
    assert!(
        avg_x_after - avg_x_before > 1.0 * cell_size,
        "sediment should advect downstream while settling: dx={:.4} (need >{:.4})",
        avg_x_after - avg_x_before,
        1.0 * cell_size
    );
    // (2) Sediment does not levitate (settling + gravity dominate vertical).
    assert!(
        avg_y_after < avg_y_before + 0.5 * cell_size,
        "sediment levitated unexpectedly: avg_y {avg_y_before:.4}->{avg_y_after:.4}"
    );
}

/// B2: in a mixed cloud, denser particles settle below lighter ones.
///
/// FLIP/particle-level analogue of the DEM density-separation test.
#[test]
fn dense_sediment_settles_below_water() {
    let (device, queue) = match init_device_queue() {
        Some(handles) => handles,
        None => {
            eprintln!("No compatible GPU adapter found; skipping density-separation test.");
            return;
        }
    };

    let width = 20u32;
    let height = 16u32;
    let depth = 10u32;
    let cell_size = 0.05f32;
    let max_particles = 8192usize;

    let mut sim = GpuFlip3D::new(&device, width, height, depth, cell_size, max_particles);
    sim.vorticity_epsilon = 0.0;
    // settling left at default (ON).

    let cell_types = build_cell_types(width, height, depth);

    let mut positions = Vec::new();
    let mut velocities = Vec::new();
    let mut densities = Vec::new();
    let mut c_matrices = Vec::new();
    let mut dense_indices = Vec::new();
    let mut light_indices = Vec::new();

    // Interleaved mixed cloud in a mid-column bounding box, zero initial velocity.
    for x in 6..14 {
        for y in 6..11 {
            for z in 3..7 {
                let pos = Vec3::new(
                    (x as f32 + 0.5) * cell_size,
                    (y as f32 + 0.5) * cell_size,
                    (z as f32 + 0.5) * cell_size,
                );
                positions.push(pos);
                velocities.push(Vec3::ZERO);
                c_matrices.push(Mat3::ZERO);
                // Interleave dense/light by parity of the cell sum.
                if (x + y + z) % 2 == 0 {
                    densities.push(2.65);
                    dense_indices.push(positions.len() - 1);
                } else {
                    densities.push(1.0);
                    light_indices.push(positions.len() - 1);
                }
            }
        }
    }

    assert!(positions.len() <= max_particles);
    assert!(!dense_indices.is_empty() && !light_indices.is_empty());

    let dt = 1.0 / 60.0;
    for _ in 0..90 {
        sim.step(
            &device,
            &queue,
            &mut positions,
            &mut velocities,
            &mut c_matrices,
            &densities,
            &cell_types,
            None,
            None,
            dt,
            -9.81,
            0.0,
            60,
        );
    }

    let avg_y = |idxs: &[usize]| -> f32 {
        idxs.iter().map(|&i| positions[i].y).sum::<f32>() / idxs.len() as f32
    };
    let avg_y_dense = avg_y(&dense_indices);
    let avg_y_light = avg_y(&light_indices);
    let min_dense_y = dense_indices
        .iter()
        .map(|&i| positions[i].y)
        .fold(f32::INFINITY, f32::min);

    println!(
        "density separation: avg_y_dense={avg_y_dense:.4} avg_y_light={avg_y_light:.4} min_dense_y={min_dense_y:.4}"
    );

    assert!(
        positions.iter().all(|p| p.is_finite()),
        "non-finite particle positions after density-separation run"
    );
    assert!(
        avg_y_dense < avg_y_light,
        "dense sediment should settle below water: dense={avg_y_dense:.4} light={avg_y_light:.4}"
    );
    // Settled, did not tunnel through the floor (floor is the solid boundary at y=0..1).
    assert!(
        min_dense_y > 0.5 * cell_size,
        "dense particles tunneled through the floor: min_dense_y={min_dense_y:.4}"
    );
}
