//! 3D MAC (Marker-and-Cell) staggered grid for incompressible fluid simulation.

use glam::Vec3;

/// Cell classification for pressure solve.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CellType {
    /// Solid obstacle (no flow)
    Solid,
    /// Contains fluid particles
    Fluid,
    /// Empty air
    #[default]
    Air,
}

/// 3D MAC grid with staggered velocities.
///
/// Velocity components are stored on cell faces:
/// - u (X-velocity) on YZ faces at x = i * dx
/// - v (Y-velocity) on XZ faces at y = j * dx
/// - w (Z-velocity) on XY faces at z = k * dx
///
/// Pressure and cell type are stored at cell centers.
pub struct Grid3D {
    /// Number of cells in X direction
    pub width: usize,
    /// Number of cells in Y direction
    pub height: usize,
    /// Number of cells in Z direction
    pub depth: usize,
    /// Size of each cell in world units
    pub cell_size: f32,

    /// U velocity (X-component) on left YZ faces
    /// Size: (width+1) * height * depth
    pub u: Vec<f32>,

    /// V velocity (Y-component) on bottom XZ faces
    /// Size: width * (height+1) * depth
    pub v: Vec<f32>,

    /// W velocity (Z-component) on back XY faces
    /// Size: width * height * (depth+1)
    pub w: Vec<f32>,

    /// Old U velocity (for FLIP delta)
    pub u_old: Vec<f32>,

    /// Old V velocity (for FLIP delta)
    pub v_old: Vec<f32>,

    /// Old W velocity (for FLIP delta)
    pub w_old: Vec<f32>,

    /// Pressure at cell centers
    pub pressure: Vec<f32>,

    /// Divergence at cell centers
    pub divergence: Vec<f32>,

    /// Cell classification (Solid/Fluid/Air)
    pub cell_type: Vec<CellType>,

    /// Permanent solid terrain
    pub solid: Vec<bool>,

    /// Signed distance field for smooth collision
    /// Positive = outside solid, Negative = inside solid
    pub sdf: Vec<f32>,
}

impl Grid3D {
    /// Create a new grid with the given dimensions.
    pub fn new(width: usize, height: usize, depth: usize, cell_size: f32) -> Self {
        let cell_count = width * height * depth;
        let u_count = (width + 1) * height * depth;
        let v_count = width * (height + 1) * depth;
        let w_count = width * height * (depth + 1);

        Self {
            width,
            height,
            depth,
            cell_size,
            u: vec![0.0; u_count],
            v: vec![0.0; v_count],
            w: vec![0.0; w_count],
            u_old: vec![0.0; u_count],
            v_old: vec![0.0; v_count],
            w_old: vec![0.0; w_count],
            pressure: vec![0.0; cell_count],
            divergence: vec![0.0; cell_count],
            cell_type: vec![CellType::Air; cell_count],
            solid: vec![false; cell_count],
            sdf: vec![f32::MAX; cell_count],
        }
    }

    /// Total world size in X direction.
    pub fn world_width(&self) -> f32 {
        self.width as f32 * self.cell_size
    }

    /// Total world size in Y direction.
    pub fn world_height(&self) -> f32 {
        self.height as f32 * self.cell_size
    }

    /// Total world size in Z direction.
    pub fn world_depth(&self) -> f32 {
        self.depth as f32 * self.cell_size
    }

    // ========== Index functions ==========

    /// Index into cell-centered arrays (pressure, cell_type, etc.)
    #[inline]
    pub fn cell_index(&self, i: usize, j: usize, k: usize) -> usize {
        k * self.width * self.height + j * self.width + i
    }

    /// Index into U velocity array (on left YZ faces).
    /// U array has dimensions (width+1) x height x depth.
    #[inline]
    pub fn u_index(&self, i: usize, j: usize, k: usize) -> usize {
        k * (self.width + 1) * self.height + j * (self.width + 1) + i
    }

    /// Index into V velocity array (on bottom XZ faces).
    /// V array has dimensions width x (height+1) x depth.
    #[inline]
    pub fn v_index(&self, i: usize, j: usize, k: usize) -> usize {
        k * self.width * (self.height + 1) + j * self.width + i
    }

    /// Index into W velocity array (on back XY faces).
    /// W array has dimensions width x height x (depth+1).
    #[inline]
    pub fn w_index(&self, i: usize, j: usize, k: usize) -> usize {
        k * self.width * self.height + j * self.width + i
    }

    // ========== World position helpers ==========

    /// World position of U velocity node at grid indices (i, j, k).
    /// U nodes are on left YZ faces: x = i*dx, y = (j+0.5)*dx, z = (k+0.5)*dx
    #[inline]
    pub fn u_position(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(
            i as f32 * self.cell_size,
            (j as f32 + 0.5) * self.cell_size,
            (k as f32 + 0.5) * self.cell_size,
        )
    }

    /// World position of V velocity node at grid indices (i, j, k).
    /// V nodes are on bottom XZ faces: x = (i+0.5)*dx, y = j*dx, z = (k+0.5)*dx
    #[inline]
    pub fn v_position(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(
            (i as f32 + 0.5) * self.cell_size,
            j as f32 * self.cell_size,
            (k as f32 + 0.5) * self.cell_size,
        )
    }

    /// World position of W velocity node at grid indices (i, j, k).
    /// W nodes are on back XY faces: x = (i+0.5)*dx, y = (j+0.5)*dx, z = k*dx
    #[inline]
    pub fn w_position(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(
            (i as f32 + 0.5) * self.cell_size,
            (j as f32 + 0.5) * self.cell_size,
            k as f32 * self.cell_size,
        )
    }

    /// World position of cell center at grid indices (i, j, k).
    #[inline]
    pub fn cell_center(&self, i: usize, j: usize, k: usize) -> Vec3 {
        Vec3::new(
            (i as f32 + 0.5) * self.cell_size,
            (j as f32 + 0.5) * self.cell_size,
            (k as f32 + 0.5) * self.cell_size,
        )
    }

    // ========== Grid coordinate helpers ==========

    /// Convert world position to cell indices (floored).
    #[inline]
    pub fn world_to_cell(&self, pos: Vec3) -> (i32, i32, i32) {
        let cell_pos = pos / self.cell_size;
        (
            cell_pos.x.floor() as i32,
            cell_pos.y.floor() as i32,
            cell_pos.z.floor() as i32,
        )
    }

    /// Check if cell indices are within bounds.
    #[inline]
    pub fn cell_in_bounds(&self, i: i32, j: i32, k: i32) -> bool {
        i >= 0
            && i < self.width as i32
            && j >= 0
            && j < self.height as i32
            && k >= 0
            && k < self.depth as i32
    }

    // ========== Solid/boundary helpers ==========

    /// Mark a cell as solid terrain.
    pub fn set_solid(&mut self, i: usize, j: usize, k: usize) {
        if i < self.width && j < self.height && k < self.depth {
            let idx = self.cell_index(i, j, k);
            self.solid[idx] = true;
            self.cell_type[idx] = CellType::Solid;
        }
    }

    /// Check if a cell is solid.
    #[inline]
    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        if i < self.width && j < self.height && k < self.depth {
            self.solid[self.cell_index(i, j, k)]
        } else {
            true // Out of bounds is treated as solid
        }
    }

    /// Check if a cell at signed indices is solid (for boundary checks).
    #[inline]
    pub fn is_solid_signed(&self, i: i32, j: i32, k: i32) -> bool {
        if i < 0 || j < 0 || k < 0 {
            return true; // Out of bounds on negative side = solid wall
        }
        if i >= self.width as i32 || j >= self.height as i32 || k >= self.depth as i32 {
            return false; // Out of bounds on positive side = open boundary
        }
        self.solid[self.cell_index(i as usize, j as usize, k as usize)]
    }

    /// Check if U-face at (i,j,k) is on a solid boundary.
    /// U-face separates cells (i-1,j,k) and (i,j,k).
    /// Returns true if we should skip this face in G2P sampling.
    #[inline]
    pub fn is_u_face_solid(&self, i: i32, j: i32, k: i32) -> bool {
        // Domain boundaries
        if i <= 0 {
            return true; // Inlet boundary - zeroed, skip
        }
        if j < 0 || j >= self.height as i32 || k < 0 || k >= self.depth as i32 {
            return true; // Out of bounds in Y or Z
        }
        if i >= self.width as i32 {
            return false; // Outlet boundary - OPEN, don't skip
        }

        // Check adjacent cells
        let left_solid = self.is_solid_signed(i - 1, j, k);
        let right_solid = self.is_solid_signed(i, j, k);
        left_solid || right_solid
    }

    /// Check if V-face at (i,j,k) is on a solid boundary.
    /// V-face separates cells (i,j-1,k) and (i,j,k).
    #[inline]
    pub fn is_v_face_solid(&self, i: i32, j: i32, k: i32) -> bool {
        if i < 0 || i >= self.width as i32 || k < 0 || k >= self.depth as i32 {
            return true; // Out of bounds in X or Z
        }
        if j <= 0 {
            return true; // Floor boundary - zeroed, skip
        }
        if j >= self.height as i32 {
            return false; // Top boundary - OPEN, don't skip
        }

        let bottom_solid = self.is_solid_signed(i, j - 1, k);
        let top_solid = self.is_solid_signed(i, j, k);
        bottom_solid || top_solid
    }

    /// Check if W-face at (i,j,k) is on a solid boundary.
    /// W-face separates cells (i,j,k-1) and (i,j,k).
    #[inline]
    pub fn is_w_face_solid(&self, i: i32, j: i32, k: i32) -> bool {
        if i < 0 || i >= self.width as i32 || j < 0 || j >= self.height as i32 {
            return true; // Out of bounds in X or Y
        }
        // Both Z boundaries are closed walls
        if k <= 0 || k >= self.depth as i32 {
            return true; // Side walls - zeroed, skip
        }

        let back_solid = self.is_solid_signed(i, j, k - 1);
        let front_solid = self.is_solid_signed(i, j, k);
        back_solid || front_solid
    }

    // ========== Reset/clear ==========

    /// Clear all velocities to zero.
    pub fn clear_velocities(&mut self) {
        self.u.fill(0.0);
        self.v.fill(0.0);
        self.w.fill(0.0);
    }

    /// Store current velocities as old (for FLIP).
    pub fn store_old_velocities(&mut self) {
        self.u_old.copy_from_slice(&self.u);
        self.v_old.copy_from_slice(&self.v);
        self.w_old.copy_from_slice(&self.w);
    }

    /// Reset all cell types to Air (except solid terrain).
    pub fn reset_cell_types(&mut self) {
        for (idx, ct) in self.cell_type.iter_mut().enumerate() {
            *ct = if self.solid[idx] {
                CellType::Solid
            } else {
                CellType::Air
            };
        }
    }

    // ========== SDF (Signed Distance Field) ==========

    /// Compute SDF using fast sweeping method.
    /// Call this after modifying solid cells.
    pub fn compute_sdf(&mut self) {
        let dx = self.cell_size;
        let inf = f32::MAX / 2.0;

        // Initialize: solid cells = -0.5*dx (inside), others = large positive
        for idx in 0..self.sdf.len() {
            self.sdf[idx] = if self.solid[idx] { -0.5 * dx } else { inf };
        }

        // Fast sweeping in 8 diagonal directions
        let sweeps: [(i32, i32, i32); 8] = [
            (1, 1, 1),
            (-1, 1, 1),
            (1, -1, 1),
            (-1, -1, 1),
            (1, 1, -1),
            (-1, 1, -1),
            (1, -1, -1),
            (-1, -1, -1),
        ];

        for _ in 0..2 {
            // Multiple passes for better convergence
            for &(di, dj, dk) in &sweeps {
                self.sweep_sdf(di, dj, dk, dx);
            }
        }
    }

    /// Single sweep pass in given direction.
    fn sweep_sdf(&mut self, di: i32, dj: i32, dk: i32, dx: f32) {
        let w = self.width as i32;
        let h = self.height as i32;
        let d = self.depth as i32;

        let i_range: Box<dyn Iterator<Item = i32>> = if di > 0 {
            Box::new(0..w)
        } else {
            Box::new((0..w).rev())
        };

        for i in i_range {
            let j_range: Box<dyn Iterator<Item = i32>> = if dj > 0 {
                Box::new(0..h)
            } else {
                Box::new((0..h).rev())
            };

            for j in j_range {
                let k_range: Box<dyn Iterator<Item = i32>> = if dk > 0 {
                    Box::new(0..d)
                } else {
                    Box::new((0..d).rev())
                };

                for k in k_range {
                    let idx = self.cell_index(i as usize, j as usize, k as usize);

                    // Skip solid cells (keep negative distance)
                    if self.solid[idx] {
                        continue;
                    }

                    let mut min_neighbor = self.sdf[idx];

                    // Check 6 neighbors
                    let neighbors = [
                        (i - 1, j, k),
                        (i + 1, j, k),
                        (i, j - 1, k),
                        (i, j + 1, k),
                        (i, j, k - 1),
                        (i, j, k + 1),
                    ];

                    for &(ni, nj, nk) in &neighbors {
                        if ni >= 0 && ni < w && nj >= 0 && nj < h && nk >= 0 && nk < d {
                            let nidx = self.cell_index(ni as usize, nj as usize, nk as usize);
                            min_neighbor = min_neighbor.min(self.sdf[nidx] + dx);
                        }
                    }

                    self.sdf[idx] = min_neighbor;
                }
            }
        }
    }

    /// Sample SDF at world position using trilinear interpolation.
    pub fn sample_sdf(&self, pos: Vec3) -> f32 {
        let dx = self.cell_size;

        // Convert to cell-centered coordinates
        let fx = pos.x / dx - 0.5;
        let fy = pos.y / dx - 0.5;
        let fz = pos.z / dx - 0.5;

        let i0 = fx.floor() as i32;
        let j0 = fy.floor() as i32;
        let k0 = fz.floor() as i32;

        let tx = fx - i0 as f32;
        let ty = fy - j0 as f32;
        let tz = fz - k0 as f32;

        // Trilinear interpolation
        let mut result = 0.0;
        for dk in 0..2 {
            for dj in 0..2 {
                for di in 0..2 {
                    let ii = i0 + di;
                    let jj = j0 + dj;
                    let kk = k0 + dk;

                    let val = self.sdf_at(ii, jj, kk);

                    let wx = if di == 0 { 1.0 - tx } else { tx };
                    let wy = if dj == 0 { 1.0 - ty } else { ty };
                    let wz = if dk == 0 { 1.0 - tz } else { tz };

                    result += val * wx * wy * wz;
                }
            }
        }

        result
    }

    /// Get SDF value at cell, treating open boundaries as air.
    fn sdf_at(&self, i: i32, j: i32, k: i32) -> f32 {
        if i >= 0
            && i < self.width as i32
            && j >= 0
            && j < self.height as i32
            && k >= 0
            && k < self.depth as i32
        {
            self.sdf[self.cell_index(i as usize, j as usize, k as usize)]
        } else if i >= self.width as i32 || j >= self.height as i32 {
            self.cell_size // Open outlet/top = air
        } else {
            -self.cell_size // Out of bounds = inside solid
        }
    }

    /// Compute SDF gradient (normal pointing away from solid).
    pub fn sdf_gradient(&self, pos: Vec3) -> Vec3 {
        let eps = self.cell_size * 0.1;
        let dx = self.sample_sdf(pos + Vec3::X * eps) - self.sample_sdf(pos - Vec3::X * eps);
        let dy = self.sample_sdf(pos + Vec3::Y * eps) - self.sample_sdf(pos - Vec3::Y * eps);
        let dz = self.sample_sdf(pos + Vec3::Z * eps) - self.sample_sdf(pos - Vec3::Z * eps);

        let grad = Vec3::new(dx, dy, dz);
        let len = grad.length();
        if len > 1e-6 {
            grad / len
        } else {
            Vec3::Y // Default to up if gradient is zero
        }
    }

    /// Clear solid terrain and reset SDF.
    pub fn clear_solids(&mut self) {
        self.solid.fill(false);
        self.sdf.fill(f32::MAX);
        self.reset_cell_types();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let grid = Grid3D::new(16, 32, 8, 0.1);
        assert_eq!(grid.width, 16);
        assert_eq!(grid.height, 32);
        assert_eq!(grid.depth, 8);
        assert_eq!(grid.cell_size, 0.1);
    }

    #[test]
    fn test_array_sizes() {
        let grid = Grid3D::new(4, 5, 6, 1.0);
        // Cell-centered: 4 * 5 * 6 = 120
        assert_eq!(grid.pressure.len(), 4 * 5 * 6);
        // U faces: (4+1) * 5 * 6 = 150
        assert_eq!(grid.u.len(), 5 * 5 * 6);
        // V faces: 4 * (5+1) * 6 = 144
        assert_eq!(grid.v.len(), 4 * 6 * 6);
        // W faces: 4 * 5 * (6+1) = 140
        assert_eq!(grid.w.len(), 4 * 5 * 7);
    }

    #[test]
    fn test_cell_index() {
        let grid = Grid3D::new(4, 5, 6, 1.0);
        // Index should be k * (w*h) + j * w + i
        assert_eq!(grid.cell_index(0, 0, 0), 0);
        assert_eq!(grid.cell_index(1, 0, 0), 1);
        assert_eq!(grid.cell_index(0, 1, 0), 4);
        assert_eq!(grid.cell_index(0, 0, 1), 20);
        assert_eq!(grid.cell_index(3, 4, 5), 5 * 20 + 4 * 4 + 3);
    }

    #[test]
    fn test_u_position() {
        let grid = Grid3D::new(4, 4, 4, 1.0);
        // U at (0,0,0) should be at (0, 0.5, 0.5)
        let pos = grid.u_position(0, 0, 0);
        assert_eq!(pos, Vec3::new(0.0, 0.5, 0.5));

        // U at (1,0,0) should be at (1, 0.5, 0.5)
        let pos = grid.u_position(1, 0, 0);
        assert_eq!(pos, Vec3::new(1.0, 0.5, 0.5));
    }

    #[test]
    fn test_v_position() {
        let grid = Grid3D::new(4, 4, 4, 1.0);
        // V at (0,0,0) should be at (0.5, 0, 0.5)
        let pos = grid.v_position(0, 0, 0);
        assert_eq!(pos, Vec3::new(0.5, 0.0, 0.5));
    }

    #[test]
    fn test_w_position() {
        let grid = Grid3D::new(4, 4, 4, 1.0);
        // W at (0,0,0) should be at (0.5, 0.5, 0)
        let pos = grid.w_position(0, 0, 0);
        assert_eq!(pos, Vec3::new(0.5, 0.5, 0.0));
    }

    #[test]
    fn test_world_to_cell() {
        let grid = Grid3D::new(10, 10, 10, 0.5);
        let (i, j, k) = grid.world_to_cell(Vec3::new(1.2, 2.3, 0.4));
        assert_eq!(i, 2); // 1.2 / 0.5 = 2.4 -> 2
        assert_eq!(j, 4); // 2.3 / 0.5 = 4.6 -> 4
        assert_eq!(k, 0); // 0.4 / 0.5 = 0.8 -> 0
    }

    #[test]
    fn test_set_solid() {
        let mut grid = Grid3D::new(4, 4, 4, 1.0);
        grid.set_solid(1, 2, 3);
        assert!(grid.is_solid(1, 2, 3));
        assert!(!grid.is_solid(0, 0, 0));
    }
}
