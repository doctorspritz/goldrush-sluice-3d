use glam::Vec3;

/// Simple heightfield terrain for excavation.
#[derive(Clone, Debug)]
pub struct Heightfield {
    pub width: usize,
    pub depth: usize,
    pub cell_size: f32,
    pub heights: Vec<f32>,
}

impl Heightfield {
    /// Create flat terrain at given height.
    pub fn new(width: usize, depth: usize, cell_size: f32, initial_height: f32) -> Self {
        Self {
            width,
            depth,
            cell_size,
            heights: vec![initial_height; width * depth],
        }
    }

    /// Get height at grid position.
    pub fn get_height(&self, x: usize, z: usize) -> f32 {
        if x < self.width && z < self.depth {
            self.heights[z * self.width + x]
        } else {
            0.0
        }
    }

    /// Set height at grid position.
    pub fn set_height(&mut self, x: usize, z: usize, height: f32) {
        if x < self.width && z < self.depth {
            self.heights[z * self.width + x] = height;
        }
    }

    /// World bounds (X/Z).
    pub fn world_size(&self) -> Vec3 {
        Vec3::new(
            self.width as f32 * self.cell_size,
            1.0,
            self.depth as f32 * self.cell_size,
        )
    }

    /// Dig at world position, returns positions to spawn particles.
    pub fn dig(&mut self, world_x: f32, world_z: f32, radius: f32, dig_depth: f32) -> Vec<Vec3> {
        let cx = (world_x / self.cell_size) as i32;
        let cz = (world_z / self.cell_size) as i32;
        let radius_cells = (radius / self.cell_size).ceil() as i32;
        let radius_cells_sq = (radius / self.cell_size).powi(2);

        let mut spawn_positions = Vec::new();

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

                let idx = z as usize * self.width + x as usize;
                let old_h = self.heights[idx];
                let new_h = (old_h - dig_depth).max(0.0);

                if new_h < old_h {
                    self.heights[idx] = new_h;

                    let world_pos = Vec3::new(
                        (x as f32 + 0.5) * self.cell_size,
                        old_h + 0.05,
                        (z as f32 + 0.5) * self.cell_size,
                    );
                    spawn_positions.push(world_pos);
                }
            }
        }

        spawn_positions
    }

    /// Simple raycast - intersect ray with heightfield.
    pub fn raycast(&self, origin: Vec3, direction: Vec3) -> Option<Vec3> {
        let dir = direction.normalize_or_zero();
        if dir.length_squared() <= f32::EPSILON {
            return None;
        }

        let step_size = self.cell_size * 0.5;
        let world_size = self.world_size();
        let max_dist = world_size.length() * 2.0 + 1.0;

        let mut t = 0.0;
        while t < max_dist {
            let p = origin + dir * t;

            if p.x >= 0.0 && p.x < world_size.x && p.z >= 0.0 && p.z < world_size.z {
                let gx = (p.x / self.cell_size) as usize;
                let gz = (p.z / self.cell_size) as usize;
                let terrain_h = self.get_height(gx, gz);

                if p.y <= terrain_h {
                    return Some(p);
                }
            }

            t += step_size;
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heightfield_creation() {
        let hf = Heightfield::new(10, 10, 0.1, 0.5);
        assert_eq!(hf.width, 10);
        assert_eq!(hf.depth, 10);
        assert!((hf.get_height(5, 5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_dig() {
        let mut hf = Heightfield::new(10, 10, 0.1, 0.5);
        let spawns = hf.dig(0.5, 0.5, 0.15, 0.1);

        assert!(!spawns.is_empty());
        assert!(hf.get_height(5, 5) < 0.5);
    }
}
