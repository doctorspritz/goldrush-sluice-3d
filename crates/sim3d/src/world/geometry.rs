//! Mesh geometry generation for terrain and water rendering.

use super::{FineRegion, TerrainMaterial, World};

impl World {
    /// Generate vertex positions and colors for terrain mesh.
    pub fn terrain_vertices(&self) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let mut positions = Vec::with_capacity(self.width * self.depth);
        let mut colors = Vec::with_capacity(self.width * self.depth);

        for z in 0..self.depth {
            for x in 0..self.width {
                let idx = self.idx(x, z);
                let height = self.ground_height(x, z);
                let sediment = self.terrain_sediment[idx];

                positions.push([
                    (x as f32 + 0.5) * self.cell_size,
                    height,
                    (z as f32 + 0.5) * self.cell_size,
                ]);

                let sediment_ratio = (sediment / 2.0).min(1.0);
                let base_color = match self.surface_material(x, z) {
                    TerrainMaterial::Dirt => [0.4, 0.3, 0.2],
                    TerrainMaterial::Gravel => [0.5, 0.5, 0.5],
                    TerrainMaterial::Sand => [0.8, 0.7, 0.5],
                    TerrainMaterial::Clay => [0.6, 0.4, 0.3],
                    TerrainMaterial::Bedrock => [0.3, 0.3, 0.35],
                };

                let sediment_color = [0.6, 0.5, 0.4];

                colors.push([
                    base_color[0] * (1.0 - sediment_ratio) + sediment_color[0] * sediment_ratio,
                    base_color[1] * (1.0 - sediment_ratio) + sediment_color[1] * sediment_ratio,
                    base_color[2] * (1.0 - sediment_ratio) + sediment_color[2] * sediment_ratio,
                ]);
            }
        }

        (positions, colors)
    }

    /// Get vertex data for water surface mesh.
    pub fn water_vertices(&self) -> (Vec<[f32; 3]>, Vec<[f32; 4]>) {
        let mut positions = Vec::new();
        let mut colors = Vec::new();

        for z in 0..self.depth {
            for x in 0..self.width {
                let depth = self.water_depth(x, z);
                if depth < 0.01 {
                    continue;
                }

                let idx = self.idx(x, z);
                let height = self.water_surface[idx];
                let turbidity = self.suspended_sediment[idx];

                positions.push([
                    (x as f32 + 0.5) * self.cell_size,
                    height,
                    (z as f32 + 0.5) * self.cell_size,
                ]);

                let alpha = (depth / 2.0).min(0.8);
                let brown = turbidity.min(0.5) * 2.0;

                colors.push([
                    0.2 + brown * 0.4,
                    0.4 + brown * 0.2,
                    0.8 - brown * 0.4,
                    alpha,
                ]);
            }
        }

        (positions, colors)
    }
}

impl FineRegion {
    /// Get terrain and water vertices for rendering the fine region.
    pub fn terrain_vertices(&self, coarse_cell_size: f32, coarse_x_min: usize, coarse_z_min: usize) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let origin_x = coarse_x_min as f32 * coarse_cell_size;
        let origin_z = coarse_z_min as f32 * coarse_cell_size;

        let mut positions = Vec::with_capacity(self.width * self.depth);
        let mut colors = Vec::with_capacity(self.width * self.depth);

        for z in 0..self.depth {
            for x in 0..self.width {
                let height = self.ground_height(x, z);

                positions.push([
                    origin_x + (x as f32 + 0.5) * self.cell_size,
                    height,
                    origin_z + (z as f32 + 0.5) * self.cell_size,
                ]);

                // Simple brown color for terrain
                colors.push([0.45, 0.35, 0.25]);
            }
        }

        (positions, colors)
    }

    /// Get water vertices for the fine region.
    pub fn water_vertices(&self, coarse_cell_size: f32, coarse_x_min: usize, coarse_z_min: usize) -> (Vec<[f32; 3]>, Vec<[f32; 4]>) {
        let origin_x = coarse_x_min as f32 * coarse_cell_size;
        let origin_z = coarse_z_min as f32 * coarse_cell_size;

        let mut positions = Vec::new();
        let mut colors = Vec::new();

        for z in 0..self.depth {
            for x in 0..self.width {
                let depth = self.water_depth(x, z);
                if depth < 0.01 {
                    continue;
                }

                let idx = self.idx(x, z);
                let height = self.water_surface[idx];
                let turbidity = self.suspended_sediment[idx];

                positions.push([
                    origin_x + (x as f32 + 0.5) * self.cell_size,
                    height,
                    origin_z + (z as f32 + 0.5) * self.cell_size,
                ]);

                let alpha = (depth / 2.0).min(0.8);
                let brown = turbidity.min(0.5) * 2.0;

                colors.push([
                    0.2 + brown * 0.4,
                    0.4 + brown * 0.2,
                    0.8 - brown * 0.4,
                    alpha,
                ]);
            }
        }

        (positions, colors)
    }
}
