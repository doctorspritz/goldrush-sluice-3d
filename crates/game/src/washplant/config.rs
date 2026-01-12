//! Washplant configuration types
//!
//! Defines the structure and composition of mineral processing plants,
//! including equipment stages and factory methods for standard configurations.

/// Configuration for a processing stage
#[derive(Debug, Clone)]
pub struct StageConfig {
    /// Unique identifier for this stage
    pub name: String,
    /// Equipment type and its configuration
    pub equipment: EquipmentConfig,
}

/// Equipment types in a washplant
#[derive(Debug, Clone, Copy)]
pub enum EquipmentType {
    Hopper,
    Grizzly,
    Shaker,
    Sluice,
}

/// Equipment configuration combining type and parameters
#[derive(Debug, Clone)]
pub enum EquipmentConfig {
    Hopper(HopperStageConfig),
    Grizzly(GrizzlyStageConfig),
    Shaker(ShakerStageConfig),
    Sluice(SluiceStageConfig),
}

/// Hopper configuration: feeds material into the system
#[derive(Debug, Clone, Copy)]
pub struct HopperStageConfig {
    /// Feed rate in kg/s
    pub feed_rate: f32,
    /// Hopper capacity in kg
    pub capacity: f32,
    /// Height in meters
    pub height: f32,
}

/// Grizzly configuration: scalps oversized material
#[derive(Debug, Clone, Copy)]
pub struct GrizzlyStageConfig {
    /// Bar spacing in mm - determines cutoff size
    pub bar_spacing: f32,
    /// Screen angle in degrees
    pub angle: f32,
    /// Length in meters
    pub length: f32,
}

/// Shaker configuration: classifies by size using vibration
#[derive(Debug, Clone)]
pub struct ShakerStageConfig {
    /// Deck count - number of classification decks
    pub deck_count: usize,
    /// Opening size of first deck in mm
    pub top_opening: f32,
    /// Vibration frequency in Hz
    pub frequency: f32,
    /// Angle in degrees
    pub angle: f32,
}

/// Sluice configuration: final gravity separation
#[derive(Debug, Clone, Copy)]
pub struct SluiceStageConfig {
    /// Length in meters
    pub length: f32,
    /// Width in meters
    pub width: f32,
    /// Angle in degrees
    pub angle: f32,
    /// Riffles height in mm
    pub riffle_height: f32,
}

/// Complete washplant configuration
#[derive(Debug, Clone)]
pub struct PlantConfig {
    /// Sequence of processing stages
    pub stages: Vec<StageConfig>,
    /// Plant name
    pub name: String,
}

impl PlantConfig {
    /// Create a standard 4-stage washplant: Hopper → Grizzly → Shaker → Sluice
    pub fn standard_4_stage() -> Self {
        PlantConfig {
            name: "Standard 4-Stage".to_string(),
            stages: vec![
                StageConfig {
                    name: "Hopper".to_string(),
                    equipment: EquipmentConfig::Hopper(HopperStageConfig {
                        feed_rate: 100.0,
                        capacity: 1000.0,
                        height: 2.5,
                    }),
                },
                StageConfig {
                    name: "Grizzly".to_string(),
                    equipment: EquipmentConfig::Grizzly(GrizzlyStageConfig {
                        bar_spacing: 50.0,
                        angle: 45.0,
                        length: 3.0,
                    }),
                },
                StageConfig {
                    name: "Shaker".to_string(),
                    equipment: EquipmentConfig::Shaker(ShakerStageConfig {
                        deck_count: 2,
                        top_opening: 10.0,
                        frequency: 1800.0,
                        angle: 20.0,
                    }),
                },
                StageConfig {
                    name: "Sluice".to_string(),
                    equipment: EquipmentConfig::Sluice(SluiceStageConfig {
                        length: 4.0,
                        width: 1.5,
                        angle: 15.0,
                        riffle_height: 30.0,
                    }),
                },
            ],
        }
    }
}
