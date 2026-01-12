# Task 1: Washplant Config Types

**File to create:** `crates/game/src/washplant/config.rs`

## Goal
Define all configuration types for washplant stages and the overall plant.

## Types to Implement

```rust
use glam::Vec3;

/// Configuration for a single processing stage
#[derive(Clone, Debug)]
pub struct StageConfig {
    /// Human-readable name (e.g., "Hopper", "Grizzly")
    pub name: &'static str,

    /// Grid dimensions
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,

    /// Cell size in meters
    pub cell_size: f32,

    /// Maximum particles for this stage
    pub max_particles: usize,

    /// Equipment type and its specific config
    pub equipment: EquipmentType,

    /// Position offset in world space (for visual layout)
    pub world_offset: Vec3,
}

/// Equipment type enum wrapping specific configs
#[derive(Clone, Debug)]
pub enum EquipmentType {
    Hopper(HopperStageConfig),
    Grizzly(GrizzlyStageConfig),
    Shaker(ShakerStageConfig),
    Sluice(SluiceStageConfig),
}

/// Hopper-specific configuration
#[derive(Clone, Debug)]
pub struct HopperStageConfig {
    pub top_width: f32,
    pub top_depth: f32,
    pub bottom_width: f32,
    pub bottom_depth: f32,
    pub wall_thickness: usize,
}

/// Grizzly (bar screen) configuration
#[derive(Clone, Debug)]
pub struct GrizzlyStageConfig {
    pub bar_spacing: usize,
    pub bar_thickness: usize,
    pub angle_deg: f32,
}

/// Shaker deck configuration
#[derive(Clone, Debug)]
pub struct ShakerStageConfig {
    pub hole_spacing: f32,
    pub hole_radius: f32,
    pub angle_deg: f32,
    pub deck_thickness: f32,
}

/// Sluice configuration (reuses existing SluiceConfig)
#[derive(Clone, Debug)]
pub struct SluiceStageConfig {
    pub floor_height_left: usize,
    pub floor_height_right: usize,
    pub riffle_spacing: usize,
    pub riffle_height: usize,
    pub riffle_thickness: usize,
    pub wall_margin: usize,
}

/// Full washplant configuration
#[derive(Clone, Debug)]
pub struct PlantConfig {
    pub stages: Vec<StageConfig>,
    /// Indices into stages vec for transfer connections
    pub connections: Vec<(usize, usize)>,
}

impl Default for PlantConfig {
    fn default() -> Self {
        Self::standard_4_stage()
    }
}

impl PlantConfig {
    /// Standard 4-stage plant: Hopper -> Grizzly -> Shaker -> Sluice
    pub fn standard_4_stage() -> Self {
        PlantConfig {
            stages: vec![
                StageConfig {
                    name: "Hopper",
                    grid_width: 40,
                    grid_height: 60,
                    grid_depth: 40,
                    cell_size: 0.05,
                    max_particles: 50_000,
                    equipment: EquipmentType::Hopper(HopperStageConfig {
                        top_width: 1.5,
                        top_depth: 1.5,
                        bottom_width: 0.4,
                        bottom_depth: 0.4,
                        wall_thickness: 2,
                    }),
                    world_offset: Vec3::ZERO,
                },
                StageConfig {
                    name: "Grizzly",
                    grid_width: 60,
                    grid_height: 40,
                    grid_depth: 50,
                    cell_size: 0.05,
                    max_particles: 80_000,
                    equipment: EquipmentType::Grizzly(GrizzlyStageConfig {
                        bar_spacing: 4,
                        bar_thickness: 2,
                        angle_deg: 15.0,
                    }),
                    world_offset: Vec3::new(3.0, -1.0, 0.0),
                },
                StageConfig {
                    name: "Shaker",
                    grid_width: 120,
                    grid_height: 50,
                    grid_depth: 60,
                    cell_size: 0.02,
                    max_particles: 150_000,
                    equipment: EquipmentType::Shaker(ShakerStageConfig {
                        hole_spacing: 0.15,
                        hole_radius: 0.025,
                        angle_deg: 12.0,
                        deck_thickness: 0.08,
                    }),
                    world_offset: Vec3::new(6.0, -2.0, 0.0),
                },
                StageConfig {
                    name: "Sluice",
                    grid_width: 200,
                    grid_height: 60,
                    grid_depth: 50,
                    cell_size: 0.01,
                    max_particles: 300_000,
                    equipment: EquipmentType::Sluice(SluiceStageConfig {
                        floor_height_left: 30,
                        floor_height_right: 4,
                        riffle_spacing: 32,
                        riffle_height: 3,
                        riffle_thickness: 2,
                        wall_margin: 8,
                    }),
                    world_offset: Vec3::new(10.0, -3.0, 0.0),
                },
            ],
            connections: vec![(0, 1), (1, 2), (2, 3)],
        }
    }
}
```

## Also create mod.rs

```rust
// crates/game/src/washplant/mod.rs
mod config;

pub use config::*;
```

## Testing
After creating, run `cargo check -p game` to verify compilation.
