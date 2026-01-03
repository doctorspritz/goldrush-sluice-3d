//! Stage scenarios and lockable metrics for sediment diagnostics.
//!
//! These scenarios are intentionally deterministic: no RNG usage, fixed spawn
//! patterns, and explicit toggles per stage. Tests and visual demos can share
//! these functions to lock behavior over time.

use crate::flip::FlipSimulation;
use crate::particle::{Particle, ParticleMaterial, ParticleState};
use crate::sluice::{create_box, create_sluice_with_mode, SluiceConfig};
use glam::Vec2;

pub const STAGE_DT: f32 = 1.0 / 60.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageMode {
    Full,
    Dry,
}

pub type StageInit = fn(&mut FlipSimulation);
pub type StageStep = fn(&mut FlipSimulation, usize);

#[derive(Clone, Copy)]
pub struct StageSpec {
    pub name: &'static str,
    pub width: usize,
    pub height: usize,
    pub cell_size: f32,
    pub steps: usize,
    pub mode: StageMode,
    pub init: StageInit,
    pub per_frame: StageStep,
}

#[derive(Clone, Copy, Debug)]
pub struct RunConfig {
    pub steps: usize,
    pub dt: f32,
    pub metrics_sample_rate: usize,
    pub divergence_sample_rate: usize,
}

impl RunConfig {
    pub fn new(steps: usize) -> Self {
        Self {
            steps,
            dt: STAGE_DT,
            metrics_sample_rate: 30,
            divergence_sample_rate: 30,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StageSummary {
    pub frames: usize,
    pub max_divergence: f32,
    pub max_speed: f32,
    pub water_count: usize,
    pub sediment_count: usize,
    pub bedload_count: usize,
    pub suspended_count: usize,
    pub gravel_count: usize,
    pub clump_count: usize,
    pub avg_sed_vx: f32,
    pub avg_sed_vy: f32,
    pub avg_sed_x: f32,
    pub avg_sed_y: f32,
    pub avg_water_speed: f32,
    pub avg_sand_y: f32,
    pub avg_gold_y: f32,
    pub avg_magnetite_y: f32,
}

pub fn run_stage(sim: &mut FlipSimulation, stage: &StageSpec, run: RunConfig) -> StageSummary {
    let mut max_divergence = 0.0f32;
    let mut max_speed = 0.0f32;
    let metrics_rate = run.metrics_sample_rate.max(1);
    let divergence_rate = run.divergence_sample_rate.max(1);

    for frame in 0..run.steps {
        (stage.per_frame)(sim, frame);

        match stage.mode {
            StageMode::Dry => sim.update_dry(run.dt),
            StageMode::Full => sim.update(run.dt),
        }

        if frame % metrics_rate == 0 {
            max_speed = max_speed.max(max_particle_speed(sim));
        }
        if frame % divergence_rate == 0 {
            sim.grid.compute_divergence();
            max_divergence = max_divergence.max(sim.grid.total_divergence());
        }
    }

    let mut summary = summarize(sim, max_divergence);
    summary.frames = run.steps;
    summary.max_speed = summary.max_speed.max(max_speed);
    summary
}

pub fn summarize(sim: &FlipSimulation, max_divergence: f32) -> StageSummary {
    let mut water_count = 0usize;
    let mut sediment_count = 0usize;
    let mut bedload_count = 0usize;
    let mut suspended_count = 0usize;
    let mut gravel_count = 0usize;

    let mut sed_vx_sum = 0.0f32;
    let mut sed_vy_sum = 0.0f32;
    let mut sed_x_sum = 0.0f32;
    let mut sed_y_sum = 0.0f32;
    let mut water_speed_sum = 0.0f32;

    let mut sand_y_sum = 0.0f32;
    let mut sand_count = 0usize;
    let mut gold_y_sum = 0.0f32;
    let mut gold_count = 0usize;
    let mut mag_y_sum = 0.0f32;
    let mut mag_count = 0usize;

    let mut max_speed = 0.0f32;

    for p in sim.particles.iter() {
        let speed = p.velocity.length();
        max_speed = max_speed.max(speed);

        if p.material == ParticleMaterial::Water {
            water_count += 1;
            water_speed_sum += speed;
            continue;
        }

        if p.material == ParticleMaterial::Gravel {
            gravel_count += 1;
        }

        if !p.is_sediment() {
            continue;
        }

        sediment_count += 1;
        sed_vx_sum += p.velocity.x;
        sed_vy_sum += p.velocity.y;
        sed_x_sum += p.position.x;
        sed_y_sum += p.position.y;

        match p.state {
            ParticleState::Bedload => bedload_count += 1,
            ParticleState::Suspended => suspended_count += 1,
        }

        match p.material {
            ParticleMaterial::Sand => {
                sand_count += 1;
                sand_y_sum += p.position.y;
            }
            ParticleMaterial::Gold => {
                gold_count += 1;
                gold_y_sum += p.position.y;
            }
            ParticleMaterial::Magnetite => {
                mag_count += 1;
                mag_y_sum += p.position.y;
            }
            _ => {}
        }
    }

    let sediment_count_f = sediment_count as f32;
    let water_count_f = water_count as f32;

    StageSummary {
        frames: 0,
        max_divergence,
        max_speed,
        water_count,
        sediment_count,
        bedload_count,
        suspended_count,
        gravel_count,
        clump_count: sim.clumps.len(),
        avg_sed_vx: if sediment_count > 0 { sed_vx_sum / sediment_count_f } else { 0.0 },
        avg_sed_vy: if sediment_count > 0 { sed_vy_sum / sediment_count_f } else { 0.0 },
        avg_sed_x: if sediment_count > 0 { sed_x_sum / sediment_count_f } else { 0.0 },
        avg_sed_y: if sediment_count > 0 { sed_y_sum / sediment_count_f } else { 0.0 },
        avg_water_speed: if water_count > 0 { water_speed_sum / water_count_f } else { 0.0 },
        avg_sand_y: if sand_count > 0 { sand_y_sum / sand_count as f32 } else { 0.0 },
        avg_gold_y: if gold_count > 0 { gold_y_sum / gold_count as f32 } else { 0.0 },
        avg_magnetite_y: if mag_count > 0 { mag_y_sum / mag_count as f32 } else { 0.0 },
    }
}

pub fn stage_by_name(name: &str) -> Option<StageSpec> {
    match name {
        "water_sluice" => Some(stage_water_sluice()),
        "dry_sand_stream" => Some(stage_dry_sand_stream()),
        "dry_gold_stream" => Some(stage_dry_gold_stream()),
        "dry_mixed_stream" => Some(stage_dry_mixed_stream()),
        "sediment_water_no_dem" => Some(stage_sediment_water_no_dem()),
        "sediment_water_dem" => Some(stage_sediment_water_dem()),
        "two_way_coupling" => Some(stage_two_way_coupling()),
        "clump_drop" => Some(stage_clump_drop()),
        _ => None,
    }
}

pub fn stage_catalog() -> Vec<StageSpec> {
    vec![
        stage_water_sluice(),
        stage_dry_sand_stream(),
        stage_dry_gold_stream(),
        stage_dry_mixed_stream(),
        stage_sand_then_gold(),
        stage_sediment_water_no_dem(),
        stage_sediment_water_dem(),
        stage_two_way_coupling(),
        stage_clump_drop(),
    ]
}

pub fn stage_water_sluice() -> StageSpec {
    STAGE_WATER_SLUICE
}

pub fn stage_dry_sand_stream() -> StageSpec {
    STAGE_DRY_SAND_STREAM
}

pub fn stage_dry_gold_stream() -> StageSpec {
    STAGE_DRY_GOLD_STREAM
}

pub fn stage_dry_mixed_stream() -> StageSpec {
    STAGE_DRY_MIXED_STREAM
}

pub fn stage_sand_then_gold() -> StageSpec {
    STAGE_SAND_THEN_GOLD
}

pub fn stage_sediment_water_no_dem() -> StageSpec {
    STAGE_SEDIMENT_WATER_NO_DEM
}

pub fn stage_sediment_water_dem() -> StageSpec {
    STAGE_SEDIMENT_WATER_DEM
}

pub fn stage_two_way_coupling() -> StageSpec {
    STAGE_TWO_WAY_COUPLING
}

pub fn stage_clump_drop() -> StageSpec {
    STAGE_CLUMP_DROP
}

const STAGE_WIDTH: usize = 256;
const STAGE_HEIGHT: usize = 128;
const STAGE_CELL_SIZE: f32 = 1.0;
const STREAM_FRAMES: usize = 180;
const WATER_INLET_X: f32 = 5.0;
const WATER_INLET_VX: f32 = 80.0;
const WATER_INLET_VY: f32 = 5.0;
const WATER_EMITTERS: usize = 4;
const EMITTER_SPACING: f32 = 3.0;

const STAGE_WATER_SLUICE: StageSpec = StageSpec {
    name: "water_sluice",
    width: STAGE_WIDTH,
    height: STAGE_HEIGHT,
    cell_size: STAGE_CELL_SIZE,
    steps: 600,
    mode: StageMode::Full,
    init: init_water_sluice,
    per_frame: step_water_sluice,
};

const STAGE_DRY_SAND_STREAM: StageSpec = StageSpec {
    name: "dry_sand_stream",
    width: 160,
    height: 120,
    cell_size: 2.0,
    steps: 360,
    mode: StageMode::Dry,
    init: init_dry_sand,
    per_frame: step_dry_sand,
};

const STAGE_DRY_GOLD_STREAM: StageSpec = StageSpec {
    name: "dry_gold_stream",
    width: 160,
    height: 120,
    cell_size: 2.0,
    steps: 360,
    mode: StageMode::Dry,
    init: init_dry_gold,
    per_frame: step_dry_gold,
};

const STAGE_DRY_MIXED_STREAM: StageSpec = StageSpec {
    name: "dry_mixed_stream",
    width: 160,
    height: 120,
    cell_size: 2.0,
    steps: 360,
    mode: StageMode::Dry,
    init: init_dry_mixed,
    per_frame: step_dry_mixed,
};

const STAGE_SAND_THEN_GOLD: StageSpec = StageSpec {
    name: "sand_then_gold",
    width: 160,
    height: 120,
    cell_size: 2.0,
    steps: 600,
    mode: StageMode::Dry,
    init: init_dry_sand,  // Same init as dry_sand
    per_frame: step_sand_then_gold,
};

const STAGE_SEDIMENT_WATER_NO_DEM: StageSpec = StageSpec {
    name: "sediment_water_no_dem",
    width: STAGE_WIDTH,
    height: STAGE_HEIGHT,
    cell_size: STAGE_CELL_SIZE,
    steps: 600,
    mode: StageMode::Full,
    init: init_sediment_water_no_dem,
    per_frame: step_sediment_water,
};

const STAGE_SEDIMENT_WATER_DEM: StageSpec = StageSpec {
    name: "sediment_water_dem",
    width: STAGE_WIDTH,
    height: STAGE_HEIGHT,
    cell_size: STAGE_CELL_SIZE,
    steps: 600,
    mode: StageMode::Full,
    init: init_sediment_water_dem,
    per_frame: step_sediment_water,
};

const STAGE_TWO_WAY_COUPLING: StageSpec = StageSpec {
    name: "two_way_coupling",
    width: STAGE_WIDTH,
    height: STAGE_HEIGHT,
    cell_size: STAGE_CELL_SIZE,
    steps: 600,
    mode: StageMode::Full,
    init: init_two_way_coupling,
    per_frame: step_two_way_coupling,
};

const STAGE_CLUMP_DROP: StageSpec = StageSpec {
    name: "clump_drop",
    width: 160,
    height: 120,
    cell_size: 2.0,
    steps: 360,
    mode: StageMode::Full,
    init: init_clump_drop,
    per_frame: step_noop,
};

fn init_water_sluice(sim: &mut FlipSimulation) {
    let config = SluiceConfig::default();
    create_sluice_with_mode(sim, &config);
    sim.use_variable_diameter = false;
}

fn step_water_sluice(sim: &mut FlipSimulation, _frame: usize) {
    let inlet_y = (sim.grid.height / 4).saturating_sub(10) as f32;
    let spacing = sim.grid.cell_size * 0.6;
    let cluster_cols = 2;
    let cluster_rows = 2;

    for i in 0..WATER_EMITTERS {
        let y = inlet_y - (i as f32 * EMITTER_SPACING);
        spawn_cluster(
            sim,
            ParticleMaterial::Water,
            Vec2::new(WATER_INLET_X, y),
            cluster_cols,
            cluster_rows,
            spacing,
            Vec2::new(WATER_INLET_VX, WATER_INLET_VY),
        );
    }
}

fn init_dry_sand(sim: &mut FlipSimulation) {
    create_box(sim);
    sim.use_variable_diameter = false;
}

fn step_dry_sand(sim: &mut FlipSimulation, frame: usize) {
    if frame >= STREAM_FRAMES {
        return;
    }
    spawn_cluster(
        sim,
        ParticleMaterial::Sand,
        dry_stream_center(sim),
        3,
        3,
        sim.grid.cell_size * 0.7,
        Vec2::ZERO,
    );
}

fn init_dry_gold(sim: &mut FlipSimulation) {
    create_box(sim);
    sim.use_variable_diameter = false;
}

fn step_dry_gold(sim: &mut FlipSimulation, frame: usize) {
    if frame >= STREAM_FRAMES {
        return;
    }
    spawn_cluster(
        sim,
        ParticleMaterial::Gold,
        dry_stream_center(sim),
        3,
        3,
        sim.grid.cell_size * 0.7,
        Vec2::ZERO,
    );
}

fn init_dry_mixed(sim: &mut FlipSimulation) {
    create_box(sim);
    sim.use_variable_diameter = false;
}

fn step_dry_mixed(sim: &mut FlipSimulation, frame: usize) {
    if frame >= STREAM_FRAMES {
        return;
    }
    let center = dry_stream_center(sim);
    let spacing = sim.grid.cell_size * 0.7;
    spawn_cluster(sim, ParticleMaterial::Sand, center, 2, 2, spacing, Vec2::ZERO);
    spawn_cluster(sim, ParticleMaterial::Magnetite, center + Vec2::new(2.0, 0.0), 2, 2, spacing, Vec2::ZERO);
    spawn_cluster(sim, ParticleMaterial::Gold, center + Vec2::new(-2.0, 0.0), 2, 2, spacing, Vec2::ZERO);
}

// Sand first, then gold - to test layering behavior
fn step_sand_then_gold(sim: &mut FlipSimulation, frame: usize) {
    let center = dry_stream_center(sim);
    let spacing = sim.grid.cell_size * 0.7;

    if frame < STREAM_FRAMES {
        // First phase: pour sand
        spawn_cluster(sim, ParticleMaterial::Sand, center, 3, 3, spacing, Vec2::ZERO);
    } else if frame < STREAM_FRAMES * 2 {
        // Second phase: pour gold on top
        spawn_cluster(sim, ParticleMaterial::Gold, center, 2, 2, spacing, Vec2::ZERO);
    }
}

fn init_sediment_water_no_dem(sim: &mut FlipSimulation) {
    let config = SluiceConfig::default();
    create_sluice_with_mode(sim, &config);
    sim.use_variable_diameter = false;
}

fn init_sediment_water_dem(sim: &mut FlipSimulation) {
    let config = SluiceConfig::default();
    create_sluice_with_mode(sim, &config);
    sim.use_variable_diameter = false;
}

fn step_sediment_water(sim: &mut FlipSimulation, frame: usize) {
    step_water_sluice(sim, frame);
    if frame % 8 == 0 {
        let inlet_y = (sim.grid.height / 4).saturating_sub(10) as f32;
        spawn_cluster(
            sim,
            ParticleMaterial::Sand,
            Vec2::new(WATER_INLET_X, inlet_y),
            1,
            1,
            sim.grid.cell_size * 0.6,
            Vec2::new(WATER_INLET_VX, WATER_INLET_VY),
        );
        spawn_cluster(
            sim,
            ParticleMaterial::Gold,
            Vec2::new(WATER_INLET_X, inlet_y),
            1,
            1,
            sim.grid.cell_size * 0.6,
            Vec2::new(WATER_INLET_VX, WATER_INLET_VY),
        );
    }
}

fn init_two_way_coupling(sim: &mut FlipSimulation) {
    let config = SluiceConfig::default();
    create_sluice_with_mode(sim, &config);
    sim.use_variable_diameter = false;
}

fn step_two_way_coupling(sim: &mut FlipSimulation, frame: usize) {
    step_water_sluice(sim, frame);
    if frame % 2 == 0 {
        let inlet_y = (sim.grid.height / 4).saturating_sub(10) as f32;
        spawn_cluster(
            sim,
            ParticleMaterial::Sand,
            Vec2::new(WATER_INLET_X, inlet_y),
            2,
            2,
            sim.grid.cell_size * 0.6,
            Vec2::new(WATER_INLET_VX, WATER_INLET_VY),
        );
    }
}

fn init_clump_drop(sim: &mut FlipSimulation) {
    create_box(sim);
    sim.use_variable_diameter = false;

    let x = sim.grid.cell_size * (sim.grid.width as f32 * 0.5);
    let y = sim.grid.cell_size * 10.0;
    // Spawn a cluster of gravel particles
    sim.spawn_gravel(x, y, 0.0, 0.0, 9);
}

fn step_noop(_sim: &mut FlipSimulation, _frame: usize) {}

fn dry_stream_center(sim: &FlipSimulation) -> Vec2 {
    let x = sim.grid.cell_size * (sim.grid.width as f32 * 0.5);
    let y = sim.grid.cell_size * 8.0;
    Vec2::new(x, y)
}

fn spawn_cluster(
    sim: &mut FlipSimulation,
    material: ParticleMaterial,
    center: Vec2,
    cols: usize,
    rows: usize,
    spacing: f32,
    velocity: Vec2,
) {
    let cols_f = cols as f32;
    let rows_f = rows as f32;
    let start_x = center.x - (cols_f - 1.0) * 0.5 * spacing;
    let start_y = center.y - (rows_f - 1.0) * 0.5 * spacing;

    for j in 0..rows {
        for i in 0..cols {
            let x = start_x + i as f32 * spacing;
            let y = start_y + j as f32 * spacing;
            spawn_direct(sim, material, x, y, velocity);
        }
    }
}

fn spawn_direct(
    sim: &mut FlipSimulation,
    material: ParticleMaterial,
    x: f32,
    y: f32,
    velocity: Vec2,
) {
    match material {
        ParticleMaterial::Water => sim.particles.spawn_water(x, y, velocity.x, velocity.y),
        ParticleMaterial::Mud => sim.particles.spawn_mud(x, y, velocity.x, velocity.y),
        ParticleMaterial::Sand => sim.particles.spawn_sand(x, y, velocity.x, velocity.y),
        ParticleMaterial::Magnetite => sim.particles.spawn_magnetite(x, y, velocity.x, velocity.y),
        ParticleMaterial::Gold => sim.particles.spawn_gold(x, y, velocity.x, velocity.y),
        ParticleMaterial::Gravel => {
            let mut p = Particle::gravel(Vec2::new(x, y), velocity);
            p.clump_id = 0;
            sim.particles.list.push(p);
        }
    }
}

fn max_particle_speed(sim: &FlipSimulation) -> f32 {
    sim.particles
        .iter()
        .map(|p| p.velocity.length())
        .fold(0.0f32, f32::max)
}
