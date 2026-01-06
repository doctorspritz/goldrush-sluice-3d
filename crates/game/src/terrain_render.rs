use sim::terrain::HeightfieldTerrain;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TerrainVertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

pub struct TerrainMesh {
    pub vertices: Vec<TerrainVertex>,
    pub indices: Vec<u32>,
}

impl TerrainMesh {
    pub fn from_heightfield(heightfield: &HeightfieldTerrain) -> Self {
        let mut vertices = Vec::with_capacity(heightfield.width * heightfield.depth * 4);
        let mut indices = Vec::with_capacity(heightfield.width * heightfield.depth * 6);

        let mut max_height: f32 = 0.0;
        for z in 0..heightfield.depth {
            for x in 0..heightfield.width {
                max_height = max_height.max(heightfield.height_at(x, z));
            }
        }

        for z in 0..heightfield.depth {
            for x in 0..heightfield.width {
                let h = heightfield.height_at(x, z);
                let color_factor = if max_height > 0.0 { h / max_height } else { 0.0 };
                let color = [
                    0.4 + 0.2 * color_factor,
                    0.25 + 0.15 * color_factor,
                    0.1 + 0.1 * color_factor,
                ];

                let x0 = x as f32 * heightfield.cell_size;
                let x1 = (x + 1) as f32 * heightfield.cell_size;
                let z0 = z as f32 * heightfield.cell_size;
                let z1 = (z + 1) as f32 * heightfield.cell_size;

                let base_idx = vertices.len() as u32;
                vertices.push(TerrainVertex { position: [x0, h, z0], color });
                vertices.push(TerrainVertex { position: [x1, h, z0], color });
                vertices.push(TerrainVertex { position: [x1, h, z1], color });
                vertices.push(TerrainVertex { position: [x0, h, z1], color });

                indices.extend_from_slice(&[
                    base_idx,
                    base_idx + 1,
                    base_idx + 2,
                    base_idx,
                    base_idx + 2,
                    base_idx + 3,
                ]);
            }
        }

        Self { vertices, indices }
    }
}
