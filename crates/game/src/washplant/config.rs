use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Configuration for a single processing stage
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageConfig {
    /// Human-readable name (e.g., "Hopper", "Grizzly")
    pub name: String,

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
    #[serde(with = "vec3_serde")]
    pub world_offset: Vec3,
}

/// Equipment type enum wrapping specific configs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EquipmentType {
    Hopper(HopperStageConfig),
    Grizzly(GrizzlyStageConfig),
    Shaker(ShakerStageConfig),
    Sluice(SluiceStageConfig),
}

/// Hopper-specific configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HopperStageConfig {
    pub top_width: f32,
    pub top_depth: f32,
    pub bottom_width: f32,
    pub bottom_depth: f32,
    pub wall_thickness: usize,
}

/// Grizzly (bar screen) configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GrizzlyStageConfig {
    pub bar_spacing: usize,
    pub bar_thickness: usize,
    pub angle_deg: f32,
}

/// Shaker deck configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShakerStageConfig {
    pub hole_spacing: f32,
    pub hole_radius: f32,
    pub angle_deg: f32,
    pub deck_thickness: f32,
    pub wall_height: usize,
    pub wall_thickness: usize,
}

/// Sluice configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SluiceStageConfig {
    pub floor_height_left: usize,
    pub floor_height_right: usize,
    pub riffle_spacing: usize,
    pub riffle_height: usize,
    pub riffle_thickness: usize,
    pub wall_margin: usize,
}

/// Transfer zone configuration between stages
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferConfig {
    /// Source stage index
    pub from_stage: usize,
    /// Destination stage index
    pub to_stage: usize,
    /// Capture zone depth in cells from end of source stage
    #[serde(default = "default_capture_depth")]
    pub capture_depth_cells: usize,
    /// Exit direction from source stage (normalized)
    #[serde(default = "default_exit_direction")]
    pub exit_direction: [f32; 3],
    /// Injection offset in destination stage (0-1 normalized coordinates)
    #[serde(default = "default_inject_offset")]
    pub inject_offset: [f32; 3],
    /// Velocity applied to particles on injection
    #[serde(default = "default_inject_velocity")]
    pub inject_velocity: [f32; 3],
    /// Time delay before particle appears in destination
    #[serde(default = "default_transit_time")]
    pub transit_time: f32,
}

fn default_capture_depth() -> usize {
    3
}
fn default_exit_direction() -> [f32; 3] {
    [1.0, 0.0, 0.0]
}
fn default_inject_offset() -> [f32; 3] {
    [0.05, 0.5, 0.5]
}
fn default_inject_velocity() -> [f32; 3] {
    [0.5, 0.0, 0.0]
}
fn default_transit_time() -> f32 {
    0.05
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            from_stage: 0,
            to_stage: 1,
            capture_depth_cells: default_capture_depth(),
            exit_direction: default_exit_direction(),
            inject_offset: default_inject_offset(),
            inject_velocity: default_inject_velocity(),
            transit_time: default_transit_time(),
        }
    }
}

impl TransferConfig {
    pub fn new(from: usize, to: usize) -> Self {
        Self {
            from_stage: from,
            to_stage: to,
            ..Default::default()
        }
    }
}

/// Full washplant configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlantConfig {
    pub stages: Vec<StageConfig>,
    /// Transfer connections between stages
    pub transfers: Vec<TransferConfig>,
}

impl Default for PlantConfig {
    fn default() -> Self {
        Self::standard_3_stage()
    }
}

impl PlantConfig {
    /// Standard 3-stage plant: Hopper -> Shaker -> Sluice (vertical stacking)
    pub fn standard_3_stage() -> Self {
        let shaker_cell_size: f32 = 0.02;
        let shaker_grid_height: usize = 60;

        PlantConfig {
            stages: vec![
                StageConfig {
                    name: "Hopper".to_string(),
                    grid_width: 20,
                    grid_height: 30,
                    grid_depth: 20,
                    cell_size: 0.03,
                    max_particles: 30_000,
                    equipment: EquipmentType::Hopper(HopperStageConfig {
                        top_width: 0.5,
                        top_depth: 0.5,
                        bottom_width: 0.25,
                        bottom_depth: 0.25,
                        wall_thickness: 2,
                    }),
                    world_offset: Vec3::new(0.15, 0.75, 0.15),
                },
                StageConfig {
                    name: "Shaker".to_string(),
                    grid_width: 120,
                    grid_height: shaker_grid_height,
                    grid_depth: 40,
                    cell_size: shaker_cell_size,
                    max_particles: 100_000,
                    equipment: EquipmentType::Shaker(ShakerStageConfig {
                        hole_spacing: 0.06,
                        hole_radius: 0.012,
                        angle_deg: 12.0,
                        deck_thickness: 0.03,
                        wall_height: 12,
                        wall_thickness: 2,
                    }),
                    world_offset: Vec3::new(0.0, 0.0, 0.0),
                },
                StageConfig {
                    name: "Sluice".to_string(),
                    grid_width: 150,
                    grid_height: 40,
                    grid_depth: 40,
                    cell_size: 0.015,
                    max_particles: 200_000,
                    equipment: EquipmentType::Sluice(SluiceStageConfig {
                        floor_height_left: 20,
                        floor_height_right: 4,
                        riffle_spacing: 20,
                        riffle_height: 3,
                        riffle_thickness: 2,
                        wall_margin: 6,
                    }),
                    world_offset: Vec3::new(-0.3, -0.5, 0.0),
                },
            ],
            transfers: vec![TransferConfig::new(0, 1), TransferConfig::new(1, 2)],
        }
    }

    /// Legacy 4-stage horizontal layout
    pub fn standard_4_stage() -> Self {
        PlantConfig {
            stages: vec![
                StageConfig {
                    name: "Hopper".to_string(),
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
                    name: "Grizzly".to_string(),
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
                    name: "Shaker".to_string(),
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
                        wall_height: 8,
                        wall_thickness: 2,
                    }),
                    world_offset: Vec3::new(6.0, -2.0, 0.0),
                },
                StageConfig {
                    name: "Sluice".to_string(),
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
            transfers: vec![
                TransferConfig::new(0, 1),
                TransferConfig::new(1, 2),
                TransferConfig::new(2, 3),
            ],
        }
    }

    /// Save configuration to JSON file
    pub fn save_json(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load configuration from JSON file
    pub fn load_json(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let config = serde_json::from_str(&json)?;
        Ok(config)
    }

    /// Save configuration to YAML file
    pub fn save_yaml(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let yaml = serde_yaml::to_string(self)?;
        std::fs::write(path, yaml)?;
        Ok(())
    }

    /// Load configuration from YAML file
    pub fn load_yaml(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let yaml = std::fs::read_to_string(path)?;
        let config = serde_yaml::from_str(&yaml)?;
        Ok(config)
    }
}

/// Custom serde module for Vec3 (glam doesn't have serde by default)
mod vec3_serde {
    use glam::Vec3;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Vec3Repr {
        x: f32,
        y: f32,
        z: f32,
    }

    pub fn serialize<S>(vec: &Vec3, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Vec3Repr {
            x: vec.x,
            y: vec.y,
            z: vec.z,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec3, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = Vec3Repr::deserialize(deserializer)?;
        Ok(Vec3::new(repr.x, repr.y, repr.z))
    }
}
