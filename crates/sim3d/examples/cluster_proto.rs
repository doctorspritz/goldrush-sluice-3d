//! 3D rigid cluster prototype (CPU).
//!
//! Run with: cargo run -p sim3d --example cluster_proto --release

use rand::{rngs::StdRng, Rng, SeedableRng};
use sim3d::{ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D, IrregularStyle3D, Vec3};

const DT: f32 = 1.0 / 120.0;
const TOTAL_TIME: f32 = 5.0;
const REPORT_EVERY: usize = 60;

fn main() {
    let bounds_min = Vec3::new(0.0, 0.0, 0.0);
    let bounds_max = Vec3::new(4.0, 3.0, 4.0);
    let mut sim = ClusterSimulation3D::new(bounds_min, bounds_max);
    sim.use_dem = true;
    sim.restitution = 0.25;
    sim.friction = 0.7;
    sim.floor_friction = 1.1;
    sim.normal_stiffness = 45_000.0;
    sim.tangential_stiffness = 20_000.0;

    let particle_radius = 0.05;
    let particle_mass = 1.0;
    let templates = [
        ClumpShape3D::Tetra,
        ClumpShape3D::Cube2,
        ClumpShape3D::Flat4,
        ClumpShape3D::Rod3,
        ClumpShape3D::Irregular {
            count: 6,
            seed: 7,
            style: IrregularStyle3D::Round,
        },
    ];

    let template_ids: Vec<usize> = templates
        .iter()
        .map(|shape| sim.add_template(ClumpTemplate3D::generate(*shape, particle_radius, particle_mass)))
        .collect();

    let mut rng = StdRng::seed_from_u64(123);
    let spawn_count = 12;
    for idx in 0..spawn_count {
        let template_idx = template_ids[idx % template_ids.len()];
        let position = Vec3::new(
            rng.gen_range(0.6..3.4),
            rng.gen_range(1.6..2.6),
            rng.gen_range(0.6..3.4),
        );
        let velocity = Vec3::new(
            rng.gen_range(-0.4..0.4),
            rng.gen_range(-0.1..0.1),
            rng.gen_range(-0.4..0.4),
        );
        sim.spawn(template_idx, position, velocity);
    }

    println!("Spawned {} clusters", sim.clumps.len());
    println!("Simulating {:.1}s at {:.0} Hz", TOTAL_TIME, 1.0 / DT);

    let steps = (TOTAL_TIME / DT) as usize;
    for step in 0..steps {
        sim.step(DT);

        if step % REPORT_EVERY == 0 || step == steps - 1 {
            let time = step as f32 * DT;
            let (min_y, max_speed) = sample_stats(&sim);
            println!(
                "t={:.2}s: clusters={}, min_y={:.3}, max_speed={:.2}",
                time,
                sim.clumps.len(),
                min_y,
                max_speed
            );
        }
    }
}

fn sample_stats(sim: &ClusterSimulation3D) -> (f32, f32) {
    let mut min_y = f32::MAX;
    let mut max_speed: f32 = 0.0;

    for clump in &sim.clumps {
        let template = &sim.templates[clump.template_idx];
        max_speed = max_speed.max(clump.velocity.length());
        for offset in &template.local_offsets {
            let pos = clump.particle_world_position(*offset);
            min_y = min_y.min(pos.y - template.particle_radius);
        }
    }

    if min_y == f32::MAX {
        min_y = 0.0;
    }

    (min_y, max_speed)
}
