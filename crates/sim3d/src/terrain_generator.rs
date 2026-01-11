//! Procedural terrain generator for Klondike-style placer deposits
//!
//! Geology model (bottom to top):
//! - Bedrock: Ancient riverbed valley carved by prehistoric river
//! - Paydirt: Gold-bearing gravel ON bedrock, thickest in valley
//! - Gravel: Sparse alluvial deposits near active creek  
//! - Overburden: Covers everything (muck/frozen soil)

use crate::World;
use noise::{Fbm, NoiseFn, Perlin};

/// Configuration for terrain generation
pub struct TerrainConfig {
    pub seed: u32,
    pub base_elevation: f32,
    pub valley_depth: f32,
    pub max_overburden: f32,
    pub min_overburden: f32,
    pub max_paydirt: f32,
    pub creek_width: f32,
}

impl Default for TerrainConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            base_elevation: 20.0, // Base bedrock height
            valley_depth: 15.0,   // How deep the valley cuts
            max_overburden: 25.0, // Maximum overburden thickness
            min_overburden: 5.0,  // Minimum overburden
            max_paydirt: 3.0,     // Maximum paydirt thickness
            creek_width: 8.0,     // Width of active creek
        }
    }
}

/// Generate Klondike-style terrain
pub fn generate_klondike_terrain(
    width: usize,
    depth: usize,
    cell_size: f32,
    config: &TerrainConfig,
) -> World {
    let mut world = World::new(width, depth, cell_size, 0.0); // Start with 0 height, we'll fill in

    // Noise generators
    let bedrock_noise: Fbm<Perlin> = Fbm::new(config.seed);
    let detail_noise: Fbm<Perlin> = Fbm::new(config.seed + 1);
    let paydirt_noise: Fbm<Perlin> = Fbm::new(config.seed + 2);
    let overburden_noise: Fbm<Perlin> = Fbm::new(config.seed + 3);

    // Valley center line (runs along Z axis with some sinusoidal + noise variation)
    let valley_center_x = (width as f32 * cell_size) / 2.0;

    for z in 0..depth {
        let mut min_ground_in_creek = 10000.0;
        let mut creek_indices = Vec::new();

        for x in 0..width {
            let idx = z * width + x;
            let world_x = x as f32 * cell_size;
            let world_z = z as f32 * cell_size;

            // 1. Calculate valley position (sinusoidal curve + noise)
            let valley_offset = (world_z * 0.01).sin() * 30.0
                + detail_noise.get([world_z as f64 * 0.005, 0.0]) as f32 * 20.0;
            let valley_x = valley_center_x + valley_offset;

            // Distance from valley center
            let dist_from_valley = (world_x - valley_x).abs();

            // 2. Bedrock elevation
            // Base elevation + large-scale noise + valley carved in
            let base = config.base_elevation
                + bedrock_noise.get([world_x as f64 * 0.003, world_z as f64 * 0.003]) as f32 * 10.0;

            // Valley profile: deeper in center, rises on sides (parabolic)
            let valley_factor = (dist_from_valley / 80.0).min(1.0); // 0 at center, 1 at 80m+ away
            let valley_carve = config.valley_depth * (1.0 - valley_factor * valley_factor);

            let bedrock = (base - valley_carve).max(2.0); // Keep minimum bedrock height
            world.bedrock_elevation[idx] = bedrock;

            // 3. Paydirt on bedrock (thicker in valley)
            let paydirt_base = config.max_paydirt * (1.0 - valley_factor * 0.8); // More in valley
            let paydirt_variation =
                paydirt_noise.get([world_x as f64 * 0.02, world_z as f64 * 0.02]) as f32;
            let paydirt = (paydirt_base + paydirt_variation * 0.5)
                .max(0.1)
                .min(config.max_paydirt);
            world.paydirt_thickness[idx] = paydirt;

            // 4. Gravel (sparse, mainly near active creek)
            let creek_dist = dist_from_valley;
            let gravel = if creek_dist < config.creek_width * 2.0 {
                let gravel_factor = 1.0 - (creek_dist / (config.creek_width * 2.0));
                gravel_factor * 1.5 // Up to 1.5m of gravel near creek
            } else {
                0.0
            };
            world.gravel_thickness[idx] = gravel;

            // 5. Overburden (covers everything, thicker on hillsides)
            let overburden_base = config.min_overburden
                + (config.max_overburden - config.min_overburden) * valley_factor;
            let overburden_variation =
                overburden_noise.get([world_x as f64 * 0.01, world_z as f64 * 0.01]) as f32;
            let overburden = (overburden_base + overburden_variation * 5.0)
                .max(config.min_overburden)
                .min(config.max_overburden);
            world.overburden_thickness[idx] = overburden;

            // 6. Creek channel: carve through overburden where water flows
            if dist_from_valley < config.creek_width {
                // Carve the creek channel
                let carve_factor = 1.0 - (dist_from_valley / config.creek_width);
                let creek_depth = carve_factor * (config.min_overburden * 0.8); // Carve most of overburden
                world.overburden_thickness[idx] = (overburden - creek_depth).max(0.5);

                // Track for water pass
                let ground_height = bedrock + paydirt + gravel + world.overburden_thickness[idx];
                if ground_height < min_ground_in_creek {
                    min_ground_in_creek = ground_height;
                }
                creek_indices.push(idx);
            }
        }

        // 7. Water Pass: Fill creek to a flat mechanical level
        let water_level = min_ground_in_creek + 0.8; // Fill 0.8m above the lowest point
        for idx in creek_indices {
            // Simply set the flat water level.
            // Where water_level < ground, it will be hidden by terrain (and my shader fix handles it).
            // Where water_level > ground, it forms a pool.
            world.water_surface[idx] = water_level;
        }
    }

    world
}

/// Get the creek emitter position (source at high Z, flows toward low Z)
pub fn get_creek_source(
    width: usize,
    depth: usize,
    cell_size: f32,
    config: &TerrainConfig,
) -> (f32, f32, f32) {
    let valley_center_x = (width as f32 * cell_size) / 2.0;
    let source_z = (depth as f32 * cell_size) * 0.9; // Near the back

    // Calculate valley offset at source
    let detail_noise: Fbm<Perlin> = Fbm::new(config.seed + 1);
    let valley_offset = (source_z * 0.01).sin() * 30.0
        + detail_noise.get([source_z as f64 * 0.005, 0.0]) as f32 * 20.0;

    let source_x = valley_center_x + valley_offset;
    let source_y = 50.0; // Approximate, will be on terrain surface

    (source_x, source_y, source_z)
}
