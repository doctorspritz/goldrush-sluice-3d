use glam::Vec3;

use crate::collapse;
use crate::collapse::CollapseEvent;
use crate::excavation::{ExcavatedParticle, Tool};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainMaterial {
    Soil,
    Sand,
    Gravel,
    Clay,
    Bedrock,
    Permafrost,
}

impl TerrainMaterial {
    pub fn angle_of_repose(&self) -> f32 {
        match self {
            Self::Sand => 30.0_f32.to_radians(),
            Self::Soil => 40.0_f32.to_radians(),
            Self::Gravel => 35.0_f32.to_radians(),
            Self::Clay => 45.0_f32.to_radians(),
            Self::Bedrock => 90.0_f32.to_radians(),
            Self::Permafrost => 70.0_f32.to_radians(),
        }
    }

    pub fn dig_hardness(&self) -> f32 {
        match self {
            Self::Sand => 1.0,
            Self::Soil => 2.0,
            Self::Gravel => 3.0,
            Self::Clay => 2.5,
            Self::Bedrock => 10.0,
            Self::Permafrost => 8.0,
        }
    }

    pub fn gold_probability(&self) -> f32 {
        match self {
            Self::Sand => 0.0,
            Self::Soil => 0.001,
            Self::Gravel => 0.05,
            Self::Clay => 0.002,
            Self::Bedrock => 0.02,
            Self::Permafrost => 0.01,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TerrainLayer {
    pub material: TerrainMaterial,
    pub thickness: f32,
    pub gold_ppm: f32,
}

/// Simple heightfield terrain with material layers.
pub struct HeightfieldTerrain {
    pub width: usize,
    pub depth: usize,
    pub cell_size: f32,
    heights: Vec<f32>,
    layers: Vec<Vec<TerrainLayer>>,
}

impl HeightfieldTerrain {
    pub fn new(width: usize, depth: usize, cell_size: f32, material: TerrainMaterial, height: f32) -> Self {
        let base_layer = TerrainLayer {
            material,
            thickness: height,
            gold_ppm: material.gold_probability() * 1_000_000.0,
        };

        let heights = vec![height; width * depth];
        let layers = vec![vec![base_layer]; width * depth];

        Self {
            width,
            depth,
            cell_size,
            heights,
            layers,
        }
    }

    pub fn height_at(&self, x: usize, z: usize) -> f32 {
        if x < self.width && z < self.depth {
            self.heights[self.column_index(x, z)]
        } else {
            0.0
        }
    }

    pub fn top_material(&self, x: usize, z: usize) -> TerrainMaterial {
        self.layers
            .get(self.column_index(x, z))
            .and_then(|layers| layers.last())
            .map(|layer| layer.material)
            .unwrap_or(TerrainMaterial::Soil)
    }

    pub fn excavate(
        &mut self,
        position: Vec3,
        radius: f32,
        depth: f32,
        tool: &Tool,
    ) -> Vec<ExcavatedParticle> {
        let dig_radius = radius.min(tool.radius);
        let dig_depth = depth.min(tool.depth);

        let cx = (position.x / self.cell_size) as i32;
        let cz = (position.z / self.cell_size) as i32;
        let radius_cells = (dig_radius / self.cell_size).ceil() as i32;
        let radius_cells_sq = (dig_radius / self.cell_size).powi(2);

        let mut excavated = Vec::new();

        for dz in -radius_cells..=radius_cells {
            for dx in -radius_cells..=radius_cells {
                let x = cx + dx;
                let z = cz + dz;

                if x < 0 || x >= self.width as i32 || z < 0 || z >= self.depth as i32 {
                    continue;
                }

                let dist_sq = (dx * dx + dz * dz) as f32;
                if dist_sq > radius_cells_sq {
                    continue;
                }

                let x = x as usize;
                let z = z as usize;
                let top_material = self.top_material(x, z);
                if tool.strength < top_material.dig_hardness() {
                    continue;
                }

                let idx = self.column_index(x, z);
                let original_height = self.heights[idx];
                let mut remaining = dig_depth;
                let mut removed_total = 0.0;

                while remaining > 0.0 {
                    let Some(mut layer) = self.layers[idx].pop() else { break };
                    let removed = remaining.min(layer.thickness);
                    layer.thickness -= removed;
                    remaining -= removed;
                    removed_total += removed;

                    if removed > 0.0 {
                        let world_pos = Vec3::new(
                            (x as f32 + 0.5) * self.cell_size,
                            (original_height - removed_total) + 0.02,
                            (z as f32 + 0.5) * self.cell_size,
                        );

                        excavated.push(ExcavatedParticle {
                            position: world_pos,
                            material: layer.material,
                            volume: removed * self.cell_size * self.cell_size,
                            gold_ppm: layer.gold_ppm,
                        });
                    }

                    if layer.thickness > 1e-4 {
                        self.layers[idx].push(layer);
                        break;
                    }
                }

                if removed_total > 0.0 {
                    self.heights[idx] = (self.heights[idx] - removed_total).max(0.0);
                }
            }
        }

        excavated
    }

    pub fn apply_collapse(&mut self) -> Vec<CollapseEvent> {
        collapse::apply_collapse(self)
    }

    pub(crate) fn column_index(&self, x: usize, z: usize) -> usize {
        z * self.width + x
    }

    pub(crate) fn neighbors(&self, x: usize, z: usize) -> Vec<(usize, usize)> {
        let mut neighbors = Vec::with_capacity(4);
        if x > 0 {
            neighbors.push((x - 1, z));
        }
        if x + 1 < self.width {
            neighbors.push((x + 1, z));
        }
        if z > 0 {
            neighbors.push((x, z - 1));
        }
        if z + 1 < self.depth {
            neighbors.push((x, z + 1));
        }
        neighbors
    }

    pub(crate) fn transfer_material(
        &mut self,
        from: (usize, usize),
        to: (usize, usize),
        amount: f32,
    ) -> f32 {
        if amount <= 0.0 {
            return 0.0;
        }

        let from_idx = self.column_index(from.0, from.1);
        let to_idx = self.column_index(to.0, to.1);

        let Some(mut layer) = self.layers[from_idx].pop() else { return 0.0 };
        let transfer_amount = amount.min(layer.thickness);
        if transfer_amount <= 0.0 {
            self.layers[from_idx].push(layer);
            return 0.0;
        }

        layer.thickness -= transfer_amount;
        if layer.thickness > 1e-4 {
            self.layers[from_idx].push(layer.clone());
        }

        self.heights[from_idx] = (self.heights[from_idx] - transfer_amount).max(0.0);
        self.heights[to_idx] += transfer_amount;

        if let Some(dest_layer) = self.layers[to_idx].last_mut() {
            if dest_layer.material == layer.material {
                dest_layer.thickness += transfer_amount;
                return transfer_amount;
            }
        }

        self.layers[to_idx].push(TerrainLayer {
            material: layer.material,
            thickness: transfer_amount,
            gold_ppm: layer.gold_ppm,
        });

        transfer_amount
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::excavation::Tool;

    #[test]
    fn test_heightfield_creation() {
        let terrain = HeightfieldTerrain::new(4, 3, 1.0, TerrainMaterial::Soil, 2.0);
        assert_eq!(terrain.height_at(2, 1), 2.0);
        assert_eq!(terrain.top_material(2, 1), TerrainMaterial::Soil);
    }

    #[test]
    fn test_excavate_respects_hardness() {
        let mut terrain = HeightfieldTerrain::new(4, 4, 1.0, TerrainMaterial::Bedrock, 2.0);
        let tool = Tool::shovel();
        let removed = terrain.excavate(Vec3::new(1.5, 0.0, 1.5), 1.0, 0.5, &tool);
        assert!(removed.is_empty());
        assert_eq!(terrain.height_at(1, 1), 2.0);
    }

    #[test]
    fn test_excavate_reduces_height() {
        let mut terrain = HeightfieldTerrain::new(4, 4, 1.0, TerrainMaterial::Soil, 2.0);
        let tool = Tool::shovel();
        let removed = terrain.excavate(Vec3::new(1.5, 0.0, 1.5), 1.0, 0.5, &tool);
        assert!(!removed.is_empty());
        assert!(terrain.height_at(1, 1) < 2.0);
    }
}
