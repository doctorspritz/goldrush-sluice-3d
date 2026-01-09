use crate::terrain::{HeightfieldTerrain, TerrainMaterial};

#[derive(Clone, Copy, Debug)]
pub struct CollapseEvent {
    pub from: (usize, usize),
    pub to: (usize, usize),
    pub amount: f32,
    pub material: TerrainMaterial,
}

/// Apply angle-of-repose collapse to a heightfield.
pub fn apply_collapse(terrain: &mut HeightfieldTerrain) -> Vec<CollapseEvent> {
    let mut events = Vec::new();
    let max_iters = terrain.width * terrain.depth;

    for _ in 0..max_iters {
        let mut changed = false;

        for x in 0..terrain.width {
            for z in 0..terrain.depth {
                let height = terrain.height_at(x, z);
                let material = terrain.top_material(x, z);
                let max_slope = material.angle_of_repose();

                for (nx, nz) in terrain.neighbors(x, z) {
                    let neighbor_height = terrain.height_at(nx, nz);
                    let height_diff = height - neighbor_height;
                    if height_diff <= 0.0 {
                        continue;
                    }

                    let slope = height_diff.atan2(terrain.cell_size);
                    if slope <= max_slope {
                        continue;
                    }

                    let transfer = terrain.cell_size * (slope - max_slope).tan() * 0.5;
                    let moved = terrain.transfer_material((x, z), (nx, nz), transfer);
                    if moved > 0.0 {
                        events.push(CollapseEvent {
                            from: (x, z),
                            to: (nx, nz),
                            amount: moved,
                            material,
                        });
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    events
}
