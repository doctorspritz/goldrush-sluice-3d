//! 3D MAC (Marker-and-Cell) staggered grid for incompressible fluid simulation.

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Cell classification for pressure solve.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Serialize, Deserialize)]
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
    u: Vec<f32>,

    /// V velocity (Y-component) on bottom XZ faces
    /// Size: width * (height+1) * depth
    v: Vec<f32>,

    /// W velocity (Z-component) on back XY faces
    /// Size: width * height * (depth+1)
    w: Vec<f32>,

    /// Old U velocity (for FLIP delta)
    u_old: Vec<f32>,

    /// Old V velocity (for FLIP delta)
    v_old: Vec<f32>,

    /// Old W velocity (for FLIP delta)
    w_old: Vec<f32>,

    /// Pressure at cell centers
    pressure: Vec<f32>,

    /// Divergence at cell centers
    divergence: Vec<f32>,

    /// Cell classification (Solid/Fluid/Air)
    cell_type: Vec<CellType>,

    /// Permanent solid terrain
    solid: Vec<bool>,

    /// Signed distance field for smooth collision
    /// Positive = outside solid, Negative = inside solid
    sdf: Vec<f32>,
}

impl Grid3D {
    /// Create a new grid with the given dimensions.
    pub fn new(width: usize, height: usize, depth: usize, cell_size: f32) -> Self {
        assert!(cell_size > 0.0, "cell_size must be positive, got {}", cell_size);
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

    // ========== Field accessors ==========

    /// Get immutable reference to U velocity field.
    pub fn u(&self) -> &[f32] {
        &self.u
    }

    /// Get mutable reference to U velocity field.
    pub fn u_mut(&mut self) -> &mut [f32] {
        &mut self.u
    }

    /// Get immutable reference to V velocity field.
    pub fn v(&self) -> &[f32] {
        &self.v
    }

    /// Get mutable reference to V velocity field.
    pub fn v_mut(&mut self) -> &mut [f32] {
        &mut self.v
    }

    /// Get immutable reference to W velocity field.
    pub fn w(&self) -> &[f32] {
        &self.w
    }

    /// Get mutable reference to W velocity field.
    pub fn w_mut(&mut self) -> &mut [f32] {
        &mut self.w
    }

    /// Get immutable reference to old U velocity field.
    pub fn u_old(&self) -> &[f32] {
        &self.u_old
    }

    /// Get mutable reference to old U velocity field.
    pub fn u_old_mut(&mut self) -> &mut [f32] {
        &mut self.u_old
    }

    /// Get immutable reference to old V velocity field.
    pub fn v_old(&self) -> &[f32] {
        &self.v_old
    }

    /// Get mutable reference to old V velocity field.
    pub fn v_old_mut(&mut self) -> &mut [f32] {
        &mut self.v_old
    }

    /// Get immutable reference to old W velocity field.
    pub fn w_old(&self) -> &[f32] {
        &self.w_old
    }

    /// Get mutable reference to old W velocity field.
    pub fn w_old_mut(&mut self) -> &mut [f32] {
        &mut self.w_old
    }

    /// Get immutable reference to pressure field.
    pub fn pressure(&self) -> &[f32] {
        &self.pressure
    }

    /// Get mutable reference to pressure field.
    pub fn pressure_mut(&mut self) -> &mut [f32] {
        &mut self.pressure
    }

    /// Get immutable reference to divergence field.
    pub fn divergence(&self) -> &[f32] {
        &self.divergence
    }

    /// Get mutable reference to divergence field.
    pub fn divergence_mut(&mut self) -> &mut [f32] {
        &mut self.divergence
    }

    /// Get immutable reference to cell type field.
    pub fn cell_type(&self) -> &[CellType] {
        &self.cell_type
    }

    /// Get mutable reference to cell type field.
    pub fn cell_type_mut(&mut self) -> &mut [CellType] {
        &mut self.cell_type
    }

    /// Get immutable reference to solid field.
    pub fn solid(&self) -> &[bool] {
        &self.solid
    }

    /// Get mutable reference to solid field.
    pub fn solid_mut(&mut self) -> &mut [bool] {
        &mut self.solid
    }

    /// Get immutable reference to SDF field.
    pub fn sdf(&self) -> &[f32] {
        &self.sdf
    }

    /// Get mutable reference to SDF field.
    pub fn sdf_mut(&mut self) -> &mut [f32] {
        &mut self.sdf
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

    /// Clear a solid cell back to air.
    pub fn clear_solid(&mut self, i: usize, j: usize, k: usize) {
        if i < self.width && j < self.height && k < self.depth {
            let idx = self.cell_index(i, j, k);
            self.solid[idx] = false;
            if self.cell_type[idx] == CellType::Solid {
                self.cell_type[idx] = CellType::Air;
            }
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
        let sentinel = (self.width + self.height + self.depth) as f32 * dx; // Max possible distance

        // Initialize: solid cells = large negative, air = large positive
        for idx in 0..self.sdf.len() {
            self.sdf[idx] = if self.solid[idx] { -sentinel } else { sentinel };
        }

        // Seeds: faces between solid and air are at 0 distance
        for k in 0..self.depth {
            for j in 0..self.height {
                for i in 0..self.width {
                    let idx = self.cell_index(i, j, k);
                    if self.solid[idx] {
                        // Check neighbors
                        let mut has_air_neighbor = false;
                        if i > 0 && !self.solid[self.cell_index(i - 1, j, k)] {
                            has_air_neighbor = true;
                        }
                        if i < self.width - 1 && !self.solid[self.cell_index(i + 1, j, k)] {
                            has_air_neighbor = true;
                        }
                        if j > 0 && !self.solid[self.cell_index(i, j - 1, k)] {
                            has_air_neighbor = true;
                        }
                        if j < self.height - 1 && !self.solid[self.cell_index(i, j + 1, k)] {
                            has_air_neighbor = true;
                        }
                        if k > 0 && !self.solid[self.cell_index(i, j, k - 1)] {
                            has_air_neighbor = true;
                        }
                        if k < self.depth - 1 && !self.solid[self.cell_index(i, j, k + 1)] {
                            has_air_neighbor = true;
                        }

                        if has_air_neighbor {
                            self.sdf[idx] = -0.5 * dx;
                        }
                    } else {
                        // Check neighbors
                        let mut has_solid_neighbor = false;
                        if i > 0 && self.solid[self.cell_index(i - 1, j, k)] {
                            has_solid_neighbor = true;
                        }
                        if i < self.width - 1 && self.solid[self.cell_index(i + 1, j, k)] {
                            has_solid_neighbor = true;
                        }
                        if j > 0 && self.solid[self.cell_index(i, j - 1, k)] {
                            has_solid_neighbor = true;
                        }
                        if j < self.height - 1 && self.solid[self.cell_index(i, j + 1, k)] {
                            has_solid_neighbor = true;
                        }
                        if k > 0 && self.solid[self.cell_index(i, j, k - 1)] {
                            has_solid_neighbor = true;
                        }
                        if k < self.depth - 1 && self.solid[self.cell_index(i, j, k + 1)] {
                            has_solid_neighbor = true;
                        }

                        if has_solid_neighbor {
                            self.sdf[idx] = 0.5 * dx;
                        }
                    }
                }
            }
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

    /// Single sweep pass in given direction using Eikonal solver.
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
                    let val = self.sdf[idx];

                    // Get minimum neighbor distance for each axis (using both directions)
                    // This implements proper upwind scheme for Eikonal solver
                    let phi_x_minus = if i > 0 {
                        self.sdf[self.cell_index((i - 1) as usize, j as usize, k as usize)]
                    } else {
                        f32::INFINITY
                    };
                    let phi_x_plus = if i < w - 1 {
                        self.sdf[self.cell_index((i + 1) as usize, j as usize, k as usize)]
                    } else {
                        f32::INFINITY
                    };
                    let phi_x = phi_x_minus.min(phi_x_plus);

                    let phi_y_minus = if j > 0 {
                        self.sdf[self.cell_index(i as usize, (j - 1) as usize, k as usize)]
                    } else {
                        f32::INFINITY
                    };
                    let phi_y_plus = if j < h - 1 {
                        self.sdf[self.cell_index(i as usize, (j + 1) as usize, k as usize)]
                    } else {
                        f32::INFINITY
                    };
                    let phi_y = phi_y_minus.min(phi_y_plus);

                    let phi_z_minus = if k > 0 {
                        self.sdf[self.cell_index(i as usize, j as usize, (k - 1) as usize)]
                    } else {
                        f32::INFINITY
                    };
                    let phi_z_plus = if k < d - 1 {
                        self.sdf[self.cell_index(i as usize, j as usize, (k + 1) as usize)]
                    } else {
                        f32::INFINITY
                    };
                    let phi_z = phi_z_minus.min(phi_z_plus);

                    if val >= 0.0 {
                        // Outside: use Eikonal solver for positive distances
                        if phi_x < f32::INFINITY || phi_y < f32::INFINITY || phi_z < f32::INFINITY {
                            let new_val = self.solve_eikonal(phi_x, phi_y, phi_z, dx);
                            self.sdf[idx] = self.sdf[idx].min(new_val);
                        }
                    } else if val < 0.0 {
                        // Inside: propagate negative distances
                        if phi_x > -f32::INFINITY || phi_y > -f32::INFINITY || phi_z > -f32::INFINITY
                        {
                            let new_val = -self.solve_eikonal(-phi_x, -phi_y, -phi_z, dx);
                            self.sdf[idx] = self.sdf[idx].max(new_val);
                        }
                    }
                }
            }
        }
    }

    /// Solve the Eikonal equation |∇φ| = 1 using upwind finite differences.
    ///
    /// Given minimum neighbor distances in each axis (phi_x, phi_y, phi_z),
    /// solves for the new distance that satisfies the Eikonal equation.
    ///
    /// Returns the solution to:
    /// (φ - phi_x)² + (φ - phi_y)² + (φ - phi_z)² = dx²
    fn solve_eikonal(&self, phi_x: f32, phi_y: f32, phi_z: f32, dx: f32) -> f32 {
        // Sort the three values: phi_a <= phi_b <= phi_c
        let (phi_a, phi_b, phi_c) = Self::sort3(phi_x, phi_y, phi_z);

        // Try 1D solution: φ = phi_a + dx
        let mut new_val = phi_a + dx;

        // Try 2D solution if 1D is too large
        if new_val > phi_b {
            // Solve (φ - phi_a)² + (φ - phi_b)² = dx²
            let sum = phi_a + phi_b;
            let diff_sq = 2.0 * dx * dx - (phi_a - phi_b).powi(2);
            if diff_sq >= 0.0 {
                new_val = (sum + diff_sq.sqrt()) / 2.0;
            }
        }

        // Try 3D solution if 2D is too large
        if new_val > phi_c {
            // Solve (φ - phi_a)² + (φ - phi_b)² + (φ - phi_c)² = dx²
            let sum = phi_a + phi_b + phi_c;
            let sum_sq = phi_a * phi_a + phi_b * phi_b + phi_c * phi_c;
            let disc = sum * sum - 3.0 * (sum_sq - dx * dx);
            if disc >= 0.0 {
                new_val = (sum + disc.sqrt()) / 3.0;
            }
        }

        new_val
    }

    /// Sort three values in ascending order.
    fn sort3(a: f32, b: f32, c: f32) -> (f32, f32, f32) {
        let (min, mid, max) = if a <= b {
            if b <= c {
                (a, b, c)
            } else if a <= c {
                (a, c, b)
            } else {
                (c, a, b)
            }
        } else {
            if a <= c {
                (b, a, c)
            } else if b <= c {
                (b, c, a)
            } else {
                (c, b, a)
            }
        };
        (min, mid, max)
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
            // Out of bounds on inlet/bottom/sides = deep inside solid
            // Use large negative value so gradient always points away from boundary
            -1.0
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

    #[test]
    #[should_panic(expected = "cell_size must be positive, got 0")]
    fn test_zero_cell_size_panics() {
        let _ = Grid3D::new(4, 4, 4, 0.0);
    }

    #[test]
    #[should_panic(expected = "cell_size must be positive, got -0.1")]
    fn test_negative_cell_size_panics() {
        let _ = Grid3D::new(4, 4, 4, -0.1);
    }

    #[test]
    fn test_sdf_euclidean_distances() {
        // Create a grid with a box obstacle
        let mut grid = Grid3D::new(20, 20, 20, 1.0);

        // Set a solid box in the center
        // Box from (8,8,8) to (11,11,11) inclusive (4x4x4 box)
        for i in 8..=11 {
            for j in 8..=11 {
                for k in 8..=11 {
                    grid.set_solid(i, j, k);
                }
            }
        }

        // Compute the SDF using the fast sweeping algorithm
        grid.compute_sdf();

        // Test key property: Euclidean distance should be shorter diagonally than Manhattan
        //
        // Compare two paths from solid box:
        // Path 1: (5,9,9) - goes (3,0,0) from edge at (8,9,9) - Manhattan dist = 3
        // Path 2: (6,7,7) - goes diagonally from corner - should be less if Euclidean

        let straight_path = grid.sdf[grid.cell_index(5, 9, 9)];
        let diagonal_path = grid.sdf[grid.cell_index(6, 7, 7)];

        println!("SDF values:");
        println!("  (5,9,9) straight 3 cells in X: {:.3}", straight_path);
        println!("  (6,7,7) diagonal path: {:.3}", diagonal_path);

        // Expected for (6,7,7):
        // Distance from (6,7,7) to nearest box corner (8,8,8): sqrt(4+1+1) = sqrt(6) ≈ 2.449
        // With surface offset ≈ 2.949
        // Expected for (5,9,9):
        // Distance in X only: 3 cells from (8,9,9), with surface ≈ 3.5

        let euclidean_dist_to_corner = ((8.0_f32 - 6.0).powi(2) + (8.0_f32 - 7.0).powi(2) + (8.0_f32 - 7.0).powi(2)).sqrt();
        println!("  Euclidean distance (6,7,7) to (8,8,8): {:.3}", euclidean_dist_to_corner);

        // The key test: verify diagonal is close to true Euclidean distance
        // Allow reasonable tolerance for discretization effects
        assert!(
            (diagonal_path - 2.0).abs() < 1.0,
            "Diagonal path should be approximately 2-3 units, got {:.3}",
            diagonal_path
        );

        // Verify it's using Euclidean not Manhattan by checking
        // that diagonal distance is significantly less than sum of offsets would suggest
        // Manhattan from (6,7,7) would treat it as 2+1+1 = 4 steps
        let manhattan_estimate = 4.5;
        assert!(
            diagonal_path < manhattan_estimate - 0.5,
            "Diagonal should be less than Manhattan estimate ({:.3}), got {:.3}",
            manhattan_estimate,
            diagonal_path
        );
    }

    #[test]
    fn test_sort3() {
        assert_eq!(Grid3D::sort3(1.0, 2.0, 3.0), (1.0, 2.0, 3.0));
        assert_eq!(Grid3D::sort3(3.0, 2.0, 1.0), (1.0, 2.0, 3.0));
        assert_eq!(Grid3D::sort3(2.0, 1.0, 3.0), (1.0, 2.0, 3.0));
        assert_eq!(Grid3D::sort3(2.0, 3.0, 1.0), (1.0, 2.0, 3.0));
        assert_eq!(Grid3D::sort3(1.0, 3.0, 2.0), (1.0, 2.0, 3.0));
        assert_eq!(Grid3D::sort3(3.0, 1.0, 2.0), (1.0, 2.0, 3.0));
        assert_eq!(Grid3D::sort3(1.0, 1.0, 1.0), (1.0, 1.0, 1.0));
    }

    #[test]
    fn test_solve_eikonal() {
        let grid = Grid3D::new(4, 4, 4, 1.0);
        let dx = 1.0;

        // Test 1D solution: one neighbor at 0, others at infinity
        // φ = phi_a + dx = 0 + 1 = 1
        let result = grid.solve_eikonal(0.0, f32::INFINITY, f32::INFINITY, dx);
        assert!((result - 1.0).abs() < 0.001, "1D: expected 1.0, got {}", result);

        // Test 2D solution: two neighbors at 0
        // Solve (φ-0)² + (φ-0)² = dx²  =>  2φ² = dx²  =>  φ = dx/sqrt(2) ≈ 0.707
        let result = grid.solve_eikonal(0.0, 0.0, f32::INFINITY, dx);
        let expected_2d = dx / 2.0_f32.sqrt();
        assert!(
            (result - expected_2d).abs() < 0.001,
            "2D: expected {:.3}, got {:.3}",
            expected_2d,
            result
        );

        // Test 3D solution: three neighbors at 0
        // Solve (φ-0)² + (φ-0)² + (φ-0)² = dx²  =>  3φ² = dx²  =>  φ = dx/sqrt(3) ≈ 0.577
        let result = grid.solve_eikonal(0.0, 0.0, 0.0, dx);
        let expected_3d = dx / 3.0_f32.sqrt();
        assert!(
            (result - expected_3d).abs() < 0.001,
            "3D: expected {:.3}, got {:.3}",
            expected_3d,
            result
        );

        // Test with non-zero neighbors: phi_a=1, phi_b=1, phi_c=inf
        // 2D solution: (φ-1)² + (φ-1)² = 1²  =>  2(φ-1)² = 1  =>  φ-1 = 1/sqrt(2)  =>  φ = 1 + 1/sqrt(2) ≈ 1.707
        let result = grid.solve_eikonal(1.0, 1.0, f32::INFINITY, dx);
        let expected = 1.0 + dx / 2.0_f32.sqrt();
        assert!(
            (result - expected).abs() < 0.001,
            "2D with offset: expected {:.3}, got {:.3}",
            expected,
            result
        );
    }
}
