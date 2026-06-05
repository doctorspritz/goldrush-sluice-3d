//! CPU 2.5D sediment-flow correctness tests.
//!
//! Deterministic, headless, always-run (no GPU). These prove sediment transport behavior
//! beyond "suspended > 0": centroid advection downstream, preferential deposition behind a
//! riffle, and mass conservation through the full erode -> advect -> settle pipeline.

use sim3d::World;

// ---------------------------------------------------------------------------
// Helpers (mirror terrain_physics.rs)
// ---------------------------------------------------------------------------

fn create_test_world(width: usize, depth: usize, cell_size: f32) -> World {
    let mut world = World::new(width, depth, cell_size, 0.0);
    world.params.open_boundaries = false; // closed system
    world
}

fn set_layer_heights(
    world: &mut World,
    bedrock: f32,
    paydirt: f32,
    gravel: f32,
    overburden: f32,
    sediment: f32,
) {
    for i in 0..world.width * world.depth {
        world.bedrock_elevation[i] = bedrock;
        world.paydirt_thickness[i] = paydirt;
        world.gravel_thickness[i] = gravel;
        world.overburden_thickness[i] = overburden;
        world.terrain_sediment[i] = sediment;
    }
}

fn add_water_with_velocity(world: &mut World, depth: f32, vel_x: f32, vel_z: f32) {
    for x in 0..world.width {
        for z in 0..world.depth {
            let idx = world.idx(x, z);
            let ground = world.ground_height(x, z);
            world.water_surface[idx] = ground + depth;
        }
    }
    for i in 0..world.water_flow_x.len() {
        world.water_flow_x[i] = vel_x;
    }
    for i in 0..world.water_flow_z.len() {
        world.water_flow_z[i] = vel_z;
    }
}

fn erode(world: &mut World, dt: f32) {
    world.update_erosion(
        dt,
        world.params.hardness_overburden,
        world.params.hardness_paydirt,
        world.params.hardness_sediment,
        world.params.hardness_gravel,
    );
}

// ---------------------------------------------------------------------------
// B3: suspended-sediment plume centroid advects downstream on a slope
// ---------------------------------------------------------------------------

/// PROVES the suspended-sediment *center of mass* moves downstream (stronger than
/// "suspended > 0"). Seeds a localized plume, drives flow in +X, and tracks the
/// concentration-weighted mean X before/after.
#[test]
fn suspended_sediment_advects_downstream_on_slope() {
    let width = 64usize;
    let depth = 16usize;
    let mut world = create_test_world(width, depth, 0.1);

    // 5% slope in +X via bedrock gradient (downstream = +X, downhill).
    let slope = 0.05;
    for z in 0..world.depth {
        for x in 0..world.width {
            let idx = world.idx(x, z);
            world.bedrock_elevation[idx] = 2.0 - (x as f32 * world.cell_size * slope);
            world.paydirt_thickness[idx] = 0.0;
            world.gravel_thickness[idx] = 0.0;
            world.overburden_thickness[idx] = 0.0;
            world.terrain_sediment[idx] = 0.2;
        }
    }

    // Seed a localized suspended plume only in x in 8..12.
    for z in 0..world.depth {
        for x in 8..12 {
            let idx = world.idx(x, z);
            world.suspended_sediment[idx] = 0.1;
        }
    }

    add_water_with_velocity(&mut world, 0.3, 1.5, 0.0);

    // Concentration-weighted mean X helper.
    let mean_x = |world: &World| -> f32 {
        let mut sum_cx = 0.0f32;
        let mut sum_c = 0.0f32;
        for z in 0..world.depth {
            for x in 0..world.width {
                let idx = world.idx(x, z);
                let c = world.suspended_sediment[idx];
                sum_cx += c * x as f32;
                sum_c += c;
            }
        }
        if sum_c > 0.0 {
            sum_cx / sum_c
        } else {
            0.0
        }
    };

    let mean_x_before = mean_x(&world);

    let dt = 0.01;
    for _ in 0..200 {
        // Keep flow alive (advection does not regenerate velocity).
        for i in 0..world.water_flow_x.len() {
            world.water_flow_x[i] = 1.5;
        }
        world.update_sediment_advection(dt);
    }

    let mean_x_after = mean_x(&world);

    println!(
        "plume centroid X: before={mean_x_before:.3} after={mean_x_after:.3} (cells)"
    );

    // Net downstream displacement of the plume centroid (>2 cells = 0.2 m).
    assert!(
        mean_x_after > mean_x_before + 2.0,
        "plume centroid should move >2 cells downstream: before={mean_x_before:.3} after={mean_x_after:.3}"
    );
    // No nonphysical teleport across the whole domain.
    assert!(
        mean_x_after < mean_x_before + 30.0,
        "plume centroid moved unphysically far: before={mean_x_before:.3} after={mean_x_after:.3}"
    );
    // No upstream leakage into the inlet column.
    let upstream_leak: f32 = (0..world.depth)
        .map(|z| world.suspended_sediment[world.idx(0, z)])
        .sum();
    assert!(
        upstream_leak < 1e-4,
        "suspended sediment leaked into the upstream edge column: {upstream_leak:.6}"
    );
}

// ---------------------------------------------------------------------------
// B4: dense sediment is captured behind a riffle (bedrock bump)
// ---------------------------------------------------------------------------

/// PROVES preferential deposition in the low-shear lee of a riffle ridge.
#[test]
fn dense_sediment_captured_behind_riffle() {
    let width = 64usize;
    let depth = 16usize;
    let mut world = create_test_world(width, depth, 0.1);

    // Flat bedrock = 1.0 except a riffle ridge bump at x in 30..32 (+8cm across full z).
    set_layer_heights(&mut world, 1.0, 0.0, 0.0, 0.0, 0.0);
    for z in 0..world.depth {
        for x in 30..32 {
            let idx = world.idx(x, z);
            world.bedrock_elevation[idx] += 0.08;
        }
    }

    // Seed uniform suspended sediment everywhere.
    for i in 0..world.suspended_sediment.len() {
        world.suspended_sediment[i] = 0.05;
    }

    add_water_with_velocity(&mut world, 0.2, 2.0, 0.0);

    let dt = 0.01;
    for _ in 0..400 {
        // Keep driving flow each step.
        for i in 0..world.water_flow_x.len() {
            world.water_flow_x[i] = 2.0;
        }
        erode(&mut world, dt);
        world.update_sediment_advection(dt);
        world.update_sediment_settling(dt);
    }

    // Lee region (immediately downstream of ridge) vs open-bed control upstream.
    let region_dep = |world: &World, x_lo: usize, x_hi: usize| -> f32 {
        let mut s = 0.0f32;
        for z in 0..world.depth {
            for x in x_lo..x_hi {
                s += world.terrain_sediment[world.idx(x, z)];
            }
        }
        s
    };
    let dep_lee = region_dep(&world, 32, 36);
    let dep_open = region_dep(&world, 10, 14);

    println!("riffle deposition: lee(x32..36)={dep_lee:.5} control(x10..14)={dep_open:.5}");

    assert!(dep_lee > 0.0, "no sediment deposited in lee of riffle (dep_lee={dep_lee:.6})");
    assert!(
        dep_lee > dep_open * 1.5,
        "sediment should preferentially deposit behind the riffle: lee={dep_lee:.5} control={dep_open:.5}"
    );
}

// ---------------------------------------------------------------------------
// B5: mass conservation through the full erode -> advect -> settle cycle
// ---------------------------------------------------------------------------

/// PROVES sediment mass conservation (<0.5%) across the FULL transport pipeline.
#[test]
fn sediment_mass_conserved_through_erode_advect_settle_cycle() {
    let mut world = create_test_world(32, 32, 0.1);
    world.params.bed_porosity = 0.0; // strict mass conservation
    set_layer_heights(&mut world, 1.0, 0.0, 0.0, 0.0, 0.1);
    // NOTE ON DEPTH: the cited mass-conservation test uses 10 m water to drive the
    // concentration/depth coupling error below 0.1%. BUT update_sediment_settling()
    // skips any cell whose per-cell concentration is < 1e-4, and at 10 m depth the
    // settled_fraction (settling_velocity*dt/depth) is ~5e-5/step. After the advection
    // phase dilutes the plume across the grid, every cell falls under that skip
    // threshold, so settling becomes a no-op (verified: pre==final suspended at 10 m).
    // That is a real interaction of the settling model with deep water, not a test bug.
    // To exercise the FULL erode->advect->settle pipeline we use a shallow channel
    // (0.3 m) where settling is active; the coupling error stays within the 0.5%
    // tolerance documented below.
    add_water_with_velocity(&mut world, 0.3, 1.0, 0.0);

    let cell_area = world.cell_size * world.cell_size;
    let total_sediment = |world: &World| -> f32 {
        let terrain: f32 = world.terrain_sediment.iter().sum::<f32>() * cell_area;
        let suspended: f32 = world
            .suspended_sediment
            .iter()
            .enumerate()
            .map(|(i, &conc)| {
                let x = i % world.width;
                let z = i / world.width;
                let ground = world.ground_height(x, z);
                let depth = (world.water_surface[i] - ground).max(0.0);
                conc * depth * cell_area
            })
            .sum::<f32>();
        terrain + suspended
    };

    let initial_total = total_sediment(&world);

    let dt = 0.01;
    // Erosion phase.
    for _ in 0..20 {
        erode(&mut world, dt);
    }
    let mid_terrain: f32 = world.terrain_sediment.iter().sum::<f32>() * cell_area;

    // Advection phase (flow kept on).
    for _ in 0..20 {
        for i in 0..world.water_flow_x.len() {
            world.water_flow_x[i] = 1.0;
        }
        world.update_sediment_advection(dt);
    }

    // Settling phase: zero the flow, let it resettle.
    for i in 0..world.water_flow_x.len() {
        world.water_flow_x[i] = 0.0;
    }
    for i in 0..world.water_flow_z.len() {
        world.water_flow_z[i] = 0.0;
    }
    let initial_suspended_total: f32 = world.suspended_sediment.iter().sum();
    for _ in 0..200 {
        world.update_sediment_settling(dt);
    }

    let final_total = total_sediment(&world);
    let final_terrain: f32 = world.terrain_sediment.iter().sum::<f32>() * cell_area;
    let final_suspended_total: f32 = world.suspended_sediment.iter().sum();

    let err = ((final_total - initial_total) / initial_total).abs();
    println!(
        "mass conservation: initial={initial_total:.6} final={final_total:.6} err={:.4}%",
        err * 100.0
    );
    println!(
        "terrain: mid={mid_terrain:.6} final={final_terrain:.6}; suspended: pre-settle={initial_suspended_total:.6} final={final_suspended_total:.6}"
    );

    assert!(
        err < 0.005,
        "sediment mass error {:.3}% exceeds 0.5% through erode->advect->settle cycle",
        err * 100.0
    );
    // Settling returned mass to bed: most resettled, terrain grew back.
    assert!(
        final_suspended_total < initial_suspended_total,
        "settling should reduce suspended sediment: pre={initial_suspended_total:.6} final={final_suspended_total:.6}"
    );
    assert!(
        final_terrain > mid_terrain,
        "settling should return mass to the bed: mid_terrain={mid_terrain:.6} final_terrain={final_terrain:.6}"
    );
}
