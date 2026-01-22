use glam::Vec3;
use sim3d::clump::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D, SdfParams};

const DT: f32 = 1.0 / 120.0;

fn main() {
    println!("=== Starting Isolated DEM Test: DEM: Floor Collision ===");

    // Use sim3d::test_geometry to match washplant_editor exactly
    use sim3d::test_geometry::{TestFloor, TestSdfGenerator};

    let width = 40;
    let height = 60;
    let depth = 40;
    let cell_size = 0.025; // 2.5cm cells, same as washplant_editor
    let grid_offset = Vec3::new(-0.5, -0.5, -0.5);

    // Replicate setup_test_floor(0.0)
    let floor_y = 0.0;
    let floor = TestFloor::with_thickness(floor_y, cell_size * 4.0); // 4 cells thick
    let mut gen = TestSdfGenerator::new(width, height, depth, cell_size, grid_offset);
    gen.add_floor(&floor);

    let sdf_data = gen.sdf;

    let sdf_params = SdfParams {
        sdf: &sdf_data,
        grid_width: width,
        grid_height: height,
        grid_depth: depth,
        cell_size,
        grid_offset,
    };

    println!(
        "MATCHING: Test SDF: floor at world y={} with {}x{}x{} grid, offset {:?}",
        floor_y, width, height, depth, grid_offset
    );

    // Setup simulation
    let mut sim =
        ClusterSimulation3D::new(Vec3::new(-10.0, -10.0, -10.0), Vec3::new(10.0, 10.0, 10.0));
    sim.gravity = Vec3::new(0.0, -9.81, 0.0);
    sim.use_dem = true; // IMPORTANT: User log implies DEM particles.

    // 25 particles at y=0.5
    // 25 particles at y=0.5
    // Constants from washplant_editor.rs
    const DEM_CLUMP_RADIUS: f32 = 0.008; // 8mm
    const DEM_SAND_DENSITY: f32 = 2650.0; // kg/m^3

    // Mass = Density * Volume
    // Volume = 4/3 * pi * r^3
    let volume = (4.0 / 3.0) * std::f32::consts::PI * DEM_CLUMP_RADIUS.powi(3);
    let particle_mass = DEM_SAND_DENSITY * volume;

    println!(
        "Physical Constants: r={}, density={}, mass={}",
        DEM_CLUMP_RADIUS, DEM_SAND_DENSITY, particle_mass
    );

    // Use Irregular shape like washplant_editor (Sand)
    use sim3d::clump::IrregularStyle3D;
    let template = ClumpTemplate3D::generate(
        ClumpShape3D::Irregular {
            count: 4,
            seed: 123,
            style: IrregularStyle3D::Sharp,
        },
        DEM_CLUMP_RADIUS,
        particle_mass,
    );
    let template_idx = sim.add_template(template);

    println!("Spawned 25 particles (Sand, Irregular-Sharp) at y=0.5+, floor at y=0");
    // Match setup_dem_test loop exactly
    for i in 0..5 {
        for j in 0..5 {
            let pos = Vec3::new(
                (i as f32 - 2.0) * 0.05,
                0.5 + (j as f32) * 0.03, // Drop from 0.5m height
                0.0,
            );
            sim.spawn(template_idx, pos, Vec3::ZERO);
        }
    }
    println!("DEM particles: {}", sim.clumps.len());

    // Run simulation
    for frame in 0..301 {
        // Step with SDF
        // Note: simulation step takes dt.
        // User log shows logs every 30 frames.

        sim.step_with_sdf(DT, &sdf_params);

        if frame % 30 == 0 {
            let mut min_y = f32::MAX;
            let mut max_y = f32::MIN;
            let mut sum_y = 0.0;
            let mut max_vel = 0.0f32;

            for clump in &sim.clumps {
                min_y = min_y.min(clump.position.y);
                max_y = max_y.max(clump.position.y);
                sum_y += clump.position.y;
                max_vel = max_vel.max(clump.velocity.length());
            }
            let avg_y = sum_y / sim.clumps.len() as f32;

            let p0 = &sim.clumps[0];
            // We need to manually calculate sdf at p0 to match log "sdf=..."
            // But we can't easily access the private methods. We'll just print kinematic state.

            println!("[Frame {}] N={}, Y: avg={:.3} min={:.3} max={:.3}, MaxVel={:.3}, Particle0: pos={:.3},{:.3},{:.3} vel={:.3},{:.3},{:.3}",
                 frame, sim.clumps.len(), avg_y, min_y, max_y, max_vel,
                 p0.position.x, p0.position.y, p0.position.z,
                 p0.velocity.x, p0.velocity.y, p0.velocity.z
             );
        }
    }
}
