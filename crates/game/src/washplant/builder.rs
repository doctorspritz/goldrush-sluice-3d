use crate::washplant::config::{
    EquipmentType, GrizzlyStageConfig, HopperStageConfig, PlantConfig, ShakerStageConfig,
    SluiceStageConfig, StageConfig, TransferConfig,
};
use glam::Vec3;
use std::path::Path;

/// Builder for constructing washplant configurations with a fluent API.
#[derive(Clone, Debug, Default)]
pub struct PlantBuilder {
    stages: Vec<StageConfig>,
    transfers: Vec<TransferConfig>,
}

impl PlantBuilder {
    /// Create a new empty plant builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start from an existing configuration.
    pub fn from_config(config: PlantConfig) -> Self {
        Self {
            stages: config.stages,
            transfers: config.transfers,
        }
    }

    /// Load configuration from a JSON file.
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let config = PlantConfig::load_json(path.as_ref())?;
        Ok(Self::from_config(config))
    }

    /// Load configuration from a YAML file.
    pub fn load_yaml(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let config = PlantConfig::load_yaml(path.as_ref())?;
        Ok(Self::from_config(config))
    }

    /// Add a raw stage configuration.
    pub fn add_stage(mut self, config: StageConfig) -> Self {
        self.stages.push(config);
        self
    }

    /// Start building a hopper stage.
    pub fn add_hopper(self, name: impl Into<String>) -> HopperBuilder {
        HopperBuilder::new(self, name.into())
    }

    /// Start building a grizzly stage.
    pub fn add_grizzly(self, name: impl Into<String>) -> GrizzlyBuilder {
        GrizzlyBuilder::new(self, name.into())
    }

    /// Start building a shaker stage.
    pub fn add_shaker(self, name: impl Into<String>) -> ShakerBuilder {
        ShakerBuilder::new(self, name.into())
    }

    /// Start building a sluice stage.
    pub fn add_sluice(self, name: impl Into<String>) -> SluiceBuilder {
        SluiceBuilder::new(self, name.into())
    }

    /// Start building a transfer connection between stages.
    pub fn connect(self, from_stage: usize, to_stage: usize) -> TransferBuilder {
        TransferBuilder::new(self, from_stage, to_stage)
    }

    /// Update the position of a stage by index.
    pub fn position_stage(mut self, idx: usize, offset: Vec3) -> Self {
        if let Some(stage) = self.stages.get_mut(idx) {
            stage.world_offset = offset;
        }
        self
    }

    /// Get the number of stages.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Build the final plant configuration.
    pub fn build(self) -> PlantConfig {
        PlantConfig {
            stages: self.stages,
            transfers: self.transfers,
        }
    }

    /// Save configuration to JSON file.
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let config = PlantConfig {
            stages: self.stages.clone(),
            transfers: self.transfers.clone(),
        };
        config.save_json(path.as_ref())
    }

    /// Save configuration to YAML file.
    pub fn save_yaml(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let config = PlantConfig {
            stages: self.stages.clone(),
            transfers: self.transfers.clone(),
        };
        config.save_yaml(path.as_ref())
    }
}

// =============================================================================
// Hopper Builder
// =============================================================================

/// Builder for hopper stage configuration.
pub struct HopperBuilder {
    parent: PlantBuilder,
    name: String,
    grid_width: usize,
    grid_height: usize,
    grid_depth: usize,
    cell_size: f32,
    max_particles: usize,
    world_offset: Vec3,
    top_width: f32,
    top_depth: f32,
    bottom_width: f32,
    bottom_depth: f32,
    wall_thickness: usize,
}

impl HopperBuilder {
    fn new(parent: PlantBuilder, name: String) -> Self {
        Self {
            parent,
            name,
            grid_width: 20,
            grid_height: 30,
            grid_depth: 20,
            cell_size: 0.03,
            max_particles: 30_000,
            world_offset: Vec3::ZERO,
            top_width: 0.5,
            top_depth: 0.5,
            bottom_width: 0.25,
            bottom_depth: 0.25,
            wall_thickness: 2,
        }
    }

    /// Set grid dimensions.
    pub fn grid(mut self, width: usize, height: usize, depth: usize) -> Self {
        self.grid_width = width;
        self.grid_height = height;
        self.grid_depth = depth;
        self
    }

    /// Set cell size in meters.
    pub fn cell_size(mut self, size: f32) -> Self {
        self.cell_size = size;
        self
    }

    /// Set maximum particle count.
    pub fn max_particles(mut self, count: usize) -> Self {
        self.max_particles = count;
        self
    }

    /// Set world position offset.
    pub fn position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.world_offset = Vec3::new(x, y, z);
        self
    }

    /// Set top opening size.
    pub fn top_size(mut self, width: f32, depth: f32) -> Self {
        self.top_width = width;
        self.top_depth = depth;
        self
    }

    /// Set bottom opening size.
    pub fn bottom_size(mut self, width: f32, depth: f32) -> Self {
        self.bottom_width = width;
        self.bottom_depth = depth;
        self
    }

    /// Set wall thickness in cells.
    pub fn wall_thickness(mut self, thickness: usize) -> Self {
        self.wall_thickness = thickness;
        self
    }

    /// Finish building this hopper and return to parent builder.
    pub fn done(mut self) -> PlantBuilder {
        let config = StageConfig {
            name: self.name,
            grid_width: self.grid_width,
            grid_height: self.grid_height,
            grid_depth: self.grid_depth,
            cell_size: self.cell_size,
            max_particles: self.max_particles,
            world_offset: self.world_offset,
            equipment: EquipmentType::Hopper(HopperStageConfig {
                top_width: self.top_width,
                top_depth: self.top_depth,
                bottom_width: self.bottom_width,
                bottom_depth: self.bottom_depth,
                wall_thickness: self.wall_thickness,
            }),
        };
        self.parent.stages.push(config);
        self.parent
    }
}

// =============================================================================
// Grizzly Builder
// =============================================================================

/// Builder for grizzly (bar screen) stage configuration.
pub struct GrizzlyBuilder {
    parent: PlantBuilder,
    name: String,
    grid_width: usize,
    grid_height: usize,
    grid_depth: usize,
    cell_size: f32,
    max_particles: usize,
    world_offset: Vec3,
    bar_spacing: usize,
    bar_thickness: usize,
    angle_deg: f32,
}

impl GrizzlyBuilder {
    fn new(parent: PlantBuilder, name: String) -> Self {
        Self {
            parent,
            name,
            grid_width: 60,
            grid_height: 40,
            grid_depth: 50,
            cell_size: 0.05,
            max_particles: 80_000,
            world_offset: Vec3::ZERO,
            bar_spacing: 4,
            bar_thickness: 2,
            angle_deg: 15.0,
        }
    }

    /// Set grid dimensions.
    pub fn grid(mut self, width: usize, height: usize, depth: usize) -> Self {
        self.grid_width = width;
        self.grid_height = height;
        self.grid_depth = depth;
        self
    }

    /// Set cell size in meters.
    pub fn cell_size(mut self, size: f32) -> Self {
        self.cell_size = size;
        self
    }

    /// Set maximum particle count.
    pub fn max_particles(mut self, count: usize) -> Self {
        self.max_particles = count;
        self
    }

    /// Set world position offset.
    pub fn position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.world_offset = Vec3::new(x, y, z);
        self
    }

    /// Set bar spacing in cells.
    pub fn bar_spacing(mut self, spacing: usize) -> Self {
        self.bar_spacing = spacing;
        self
    }

    /// Set bar thickness in cells.
    pub fn bar_thickness(mut self, thickness: usize) -> Self {
        self.bar_thickness = thickness;
        self
    }

    /// Set grizzly angle in degrees.
    pub fn angle(mut self, deg: f32) -> Self {
        self.angle_deg = deg;
        self
    }

    /// Finish building this grizzly and return to parent builder.
    pub fn done(mut self) -> PlantBuilder {
        let config = StageConfig {
            name: self.name,
            grid_width: self.grid_width,
            grid_height: self.grid_height,
            grid_depth: self.grid_depth,
            cell_size: self.cell_size,
            max_particles: self.max_particles,
            world_offset: self.world_offset,
            equipment: EquipmentType::Grizzly(GrizzlyStageConfig {
                bar_spacing: self.bar_spacing,
                bar_thickness: self.bar_thickness,
                angle_deg: self.angle_deg,
            }),
        };
        self.parent.stages.push(config);
        self.parent
    }
}

// =============================================================================
// Shaker Builder
// =============================================================================

/// Builder for shaker deck stage configuration.
pub struct ShakerBuilder {
    parent: PlantBuilder,
    name: String,
    grid_width: usize,
    grid_height: usize,
    grid_depth: usize,
    cell_size: f32,
    max_particles: usize,
    world_offset: Vec3,
    hole_spacing: f32,
    hole_radius: f32,
    angle_deg: f32,
    deck_thickness: f32,
    wall_height: usize,
    wall_thickness: usize,
}

impl ShakerBuilder {
    fn new(parent: PlantBuilder, name: String) -> Self {
        Self {
            parent,
            name,
            grid_width: 120,
            grid_height: 60,
            grid_depth: 40,
            cell_size: 0.02,
            max_particles: 100_000,
            world_offset: Vec3::ZERO,
            hole_spacing: 0.06,
            hole_radius: 0.012,
            angle_deg: 12.0,
            deck_thickness: 0.03,
            wall_height: 12,
            wall_thickness: 2,
        }
    }

    /// Set grid dimensions.
    pub fn grid(mut self, width: usize, height: usize, depth: usize) -> Self {
        self.grid_width = width;
        self.grid_height = height;
        self.grid_depth = depth;
        self
    }

    /// Set cell size in meters.
    pub fn cell_size(mut self, size: f32) -> Self {
        self.cell_size = size;
        self
    }

    /// Set maximum particle count.
    pub fn max_particles(mut self, count: usize) -> Self {
        self.max_particles = count;
        self
    }

    /// Set world position offset.
    pub fn position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.world_offset = Vec3::new(x, y, z);
        self
    }

    /// Set hole spacing in meters.
    pub fn hole_spacing(mut self, spacing: f32) -> Self {
        self.hole_spacing = spacing;
        self
    }

    /// Set hole radius in meters.
    pub fn hole_radius(mut self, radius: f32) -> Self {
        self.hole_radius = radius;
        self
    }

    /// Set deck angle in degrees.
    pub fn angle(mut self, deg: f32) -> Self {
        self.angle_deg = deg;
        self
    }

    /// Set deck thickness in meters.
    pub fn deck_thickness(mut self, thickness: f32) -> Self {
        self.deck_thickness = thickness;
        self
    }

    /// Set wall height in cells.
    pub fn wall_height(mut self, height: usize) -> Self {
        self.wall_height = height;
        self
    }

    /// Set wall thickness in cells.
    pub fn wall_thickness(mut self, thickness: usize) -> Self {
        self.wall_thickness = thickness;
        self
    }

    /// Finish building this shaker and return to parent builder.
    pub fn done(mut self) -> PlantBuilder {
        let config = StageConfig {
            name: self.name,
            grid_width: self.grid_width,
            grid_height: self.grid_height,
            grid_depth: self.grid_depth,
            cell_size: self.cell_size,
            max_particles: self.max_particles,
            world_offset: self.world_offset,
            equipment: EquipmentType::Shaker(ShakerStageConfig {
                hole_spacing: self.hole_spacing,
                hole_radius: self.hole_radius,
                angle_deg: self.angle_deg,
                deck_thickness: self.deck_thickness,
                wall_height: self.wall_height,
                wall_thickness: self.wall_thickness,
            }),
        };
        self.parent.stages.push(config);
        self.parent
    }
}

// =============================================================================
// Sluice Builder
// =============================================================================

/// Builder for sluice stage configuration.
pub struct SluiceBuilder {
    parent: PlantBuilder,
    name: String,
    grid_width: usize,
    grid_height: usize,
    grid_depth: usize,
    cell_size: f32,
    max_particles: usize,
    world_offset: Vec3,
    floor_height_left: usize,
    floor_height_right: usize,
    riffle_spacing: usize,
    riffle_height: usize,
    riffle_thickness: usize,
    wall_margin: usize,
}

impl SluiceBuilder {
    fn new(parent: PlantBuilder, name: String) -> Self {
        Self {
            parent,
            name,
            grid_width: 150,
            grid_height: 40,
            grid_depth: 40,
            cell_size: 0.015,
            max_particles: 200_000,
            world_offset: Vec3::ZERO,
            floor_height_left: 20,
            floor_height_right: 4,
            riffle_spacing: 20,
            riffle_height: 3,
            riffle_thickness: 2,
            wall_margin: 6,
        }
    }

    /// Set grid dimensions.
    pub fn grid(mut self, width: usize, height: usize, depth: usize) -> Self {
        self.grid_width = width;
        self.grid_height = height;
        self.grid_depth = depth;
        self
    }

    /// Set cell size in meters.
    pub fn cell_size(mut self, size: f32) -> Self {
        self.cell_size = size;
        self
    }

    /// Set maximum particle count.
    pub fn max_particles(mut self, count: usize) -> Self {
        self.max_particles = count;
        self
    }

    /// Set world position offset.
    pub fn position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.world_offset = Vec3::new(x, y, z);
        self
    }

    /// Set floor heights (left is upstream, right is downstream).
    pub fn floor_heights(mut self, left: usize, right: usize) -> Self {
        self.floor_height_left = left;
        self.floor_height_right = right;
        self
    }

    /// Set riffle spacing in cells.
    pub fn riffle_spacing(mut self, spacing: usize) -> Self {
        self.riffle_spacing = spacing;
        self
    }

    /// Set riffle height in cells.
    pub fn riffle_height(mut self, height: usize) -> Self {
        self.riffle_height = height;
        self
    }

    /// Set riffle thickness in cells.
    pub fn riffle_thickness(mut self, thickness: usize) -> Self {
        self.riffle_thickness = thickness;
        self
    }

    /// Set wall margin in cells.
    pub fn wall_margin(mut self, margin: usize) -> Self {
        self.wall_margin = margin;
        self
    }

    /// Finish building this sluice and return to parent builder.
    pub fn done(mut self) -> PlantBuilder {
        let config = StageConfig {
            name: self.name,
            grid_width: self.grid_width,
            grid_height: self.grid_height,
            grid_depth: self.grid_depth,
            cell_size: self.cell_size,
            max_particles: self.max_particles,
            world_offset: self.world_offset,
            equipment: EquipmentType::Sluice(SluiceStageConfig {
                floor_height_left: self.floor_height_left,
                floor_height_right: self.floor_height_right,
                riffle_spacing: self.riffle_spacing,
                riffle_height: self.riffle_height,
                riffle_thickness: self.riffle_thickness,
                wall_margin: self.wall_margin,
            }),
        };
        self.parent.stages.push(config);
        self.parent
    }
}

// =============================================================================
// Transfer Builder
// =============================================================================

/// Builder for transfer zone configuration between stages.
pub struct TransferBuilder {
    parent: PlantBuilder,
    from_stage: usize,
    to_stage: usize,
    capture_depth_cells: usize,
    exit_direction: [f32; 3],
    inject_offset: [f32; 3],
    inject_velocity: [f32; 3],
    transit_time: f32,
}

impl TransferBuilder {
    fn new(parent: PlantBuilder, from_stage: usize, to_stage: usize) -> Self {
        Self {
            parent,
            from_stage,
            to_stage,
            capture_depth_cells: 3,
            exit_direction: [1.0, 0.0, 0.0],
            inject_offset: [0.05, 0.5, 0.5],
            inject_velocity: [0.5, 0.0, 0.0],
            transit_time: 0.05,
        }
    }

    /// Set capture zone depth in cells from end of source stage.
    pub fn capture_depth(mut self, cells: usize) -> Self {
        self.capture_depth_cells = cells;
        self
    }

    /// Set exit direction from source stage (will be normalized).
    pub fn exit_direction(mut self, x: f32, y: f32, z: f32) -> Self {
        self.exit_direction = [x, y, z];
        self
    }

    /// Set injection offset in destination stage (0-1 normalized coordinates).
    pub fn inject_offset(mut self, x: f32, y: f32, z: f32) -> Self {
        self.inject_offset = [x, y, z];
        self
    }

    /// Set velocity applied to particles on injection.
    pub fn inject_velocity(mut self, x: f32, y: f32, z: f32) -> Self {
        self.inject_velocity = [x, y, z];
        self
    }

    /// Set transit time delay before particle appears in destination.
    pub fn transit_time(mut self, time: f32) -> Self {
        self.transit_time = time;
        self
    }

    /// Finish building this transfer and return to parent builder.
    pub fn done(mut self) -> PlantBuilder {
        let config = TransferConfig {
            from_stage: self.from_stage,
            to_stage: self.to_stage,
            capture_depth_cells: self.capture_depth_cells,
            exit_direction: self.exit_direction,
            inject_offset: self.inject_offset,
            inject_velocity: self.inject_velocity,
            transit_time: self.transit_time,
        };
        self.parent.transfers.push(config);
        self.parent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_builder() {
        let config = PlantBuilder::new()
            .add_hopper("Feed Hopper")
            .top_size(1.0, 1.0)
            .bottom_size(0.3, 0.3)
            .position(0.0, 1.0, 0.0)
            .done()
            .add_shaker("Screen Deck")
            .angle(12.0)
            .hole_spacing(0.06)
            .position(0.0, 0.0, 0.0)
            .done()
            .add_sluice("Recovery Sluice")
            .riffle_spacing(20)
            .position(-0.3, -0.5, 0.0)
            .done()
            .connect(0, 1)
            .capture_depth(5)
            .inject_velocity(0.3, -0.1, 0.0)
            .done()
            .connect(1, 2)
            .transit_time(0.1)
            .done()
            .build();

        assert_eq!(config.stages.len(), 3);
        assert_eq!(config.transfers.len(), 2);
        assert_eq!(config.stages[0].name, "Feed Hopper");
        assert_eq!(config.stages[1].name, "Screen Deck");
        assert_eq!(config.stages[2].name, "Recovery Sluice");
    }

    #[test]
    fn test_stage_count() {
        let builder = PlantBuilder::new()
            .add_hopper("H1")
            .done()
            .add_shaker("S1")
            .done();

        assert_eq!(builder.stage_count(), 2);
    }
}
