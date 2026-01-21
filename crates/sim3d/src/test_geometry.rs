//! Test Geometry for Isolated Physics Testing
//!
//! Simple geometric primitives that generate SDF values for testing specific
//! physics behaviors in isolation:
//! - TestFloor: Thick horizontal slab for drop/bounce tests
//! - TestBox: Open-top container with thick walls for settling/containment tests
//! - TestRamp: Angled slab for friction/flow tests
//! - TestChute: Angled channel with thick walls for flow tests
//!
//! All geometry has thickness (not infinitely thin planes) to ensure proper
//! SDF collision detection.

use glam::Vec3;

/// Test floor - thick horizontal slab
/// The floor surface is at `y`, solid extends down to `y - thickness`
#[derive(Debug, Clone)]
pub struct TestFloor {
    pub y: f32,         // Top surface Y coordinate
    pub thickness: f32, // How thick the floor is (extends downward)
}

impl TestFloor {
    pub fn new(y: f32) -> Self {
        Self {
            y,
            thickness: 0.5, // Default 50cm thick
        }
    }

    pub fn with_thickness(y: f32, thickness: f32) -> Self {
        Self { y, thickness }
    }

    /// SDF value at a point (negative = inside solid, positive = outside)
    pub fn sdf(&self, point: Vec3) -> f32 {
        // Floor slab extends from y down to y-thickness
        // Above floor surface: positive distance
        // Inside floor slab: negative (distance to nearest surface)
        // Below floor bottom: positive distance to bottom surface

        let floor_top = self.y;
        let floor_bottom = self.y - self.thickness;

        if point.y >= floor_top {
            // Above floor - return distance to top surface
            point.y - floor_top
        } else if point.y <= floor_bottom {
            // Below floor bottom - return distance to bottom surface
            floor_bottom - point.y
        } else {
            // Inside floor slab - negative value
            // Return distance to nearest surface (as negative)
            let dist_to_top = floor_top - point.y;
            let dist_to_bottom = point.y - floor_bottom;
            -dist_to_top.min(dist_to_bottom)
        }
    }
}

/// Test box - open-top container with thick walls and floor
/// All walls and floor have configurable thickness for proper SDF collision
#[derive(Debug, Clone)]
pub struct TestBox {
    pub center: Vec3,         // Center of the box interior floor surface
    pub half_width: f32,      // Half inner width in X
    pub half_depth: f32,      // Half inner depth in Z
    pub wall_height: f32,     // Height of walls above floor
    pub wall_thickness: f32,  // Thickness of walls
    pub floor_thickness: f32, // Thickness of floor
}

impl TestBox {
    pub fn new(center: Vec3, width: f32, depth: f32, wall_height: f32) -> Self {
        Self {
            center,
            half_width: width / 2.0,
            half_depth: depth / 2.0,
            wall_height,
            wall_thickness: 0.1,  // 10cm thick walls
            floor_thickness: 0.2, // 20cm thick floor
        }
    }

    pub fn with_thickness(
        center: Vec3,
        width: f32,
        depth: f32,
        wall_height: f32,
        wall_thickness: f32,
        floor_thickness: f32,
    ) -> Self {
        Self {
            center,
            half_width: width / 2.0,
            half_depth: depth / 2.0,
            wall_height,
            wall_thickness,
            floor_thickness,
        }
    }

    /// SDF value at a point
    /// Negative = inside solid (walls/floor), Positive = in open space
    pub fn sdf(&self, point: Vec3) -> f32 {
        let local = point - self.center;

        // Inner boundary (where open space ends)
        let inner_x = self.half_width;
        let inner_z = self.half_depth;

        // Outer boundary (where solid ends)
        let outer_x = self.half_width + self.wall_thickness;
        let outer_z = self.half_depth + self.wall_thickness;

        // Floor: surface at y=0, solid extends down
        let floor_top = 0.0;
        let floor_bottom = -self.floor_thickness;

        // Check if inside the open interior
        let in_interior_x = local.x.abs() < inner_x;
        let in_interior_z = local.z.abs() < inner_z;
        let above_floor = local.y > floor_top;
        let _below_ceiling = local.y < self.wall_height; // Open top, but walls have height

        // Distance to inner surfaces (positive = in open space)
        let dist_to_floor = local.y - floor_top;
        let dist_to_wall_x = inner_x - local.x.abs();
        let dist_to_wall_z = inner_z - local.z.abs();

        // Inside open interior
        if in_interior_x && in_interior_z && above_floor {
            return dist_to_floor.min(dist_to_wall_x).min(dist_to_wall_z);
        }

        // In the floor slab
        if local.y <= floor_top && local.y >= floor_bottom {
            if local.x.abs() <= outer_x && local.z.abs() <= outer_z {
                // Inside floor solid
                let to_top = local.y - floor_top;
                let to_bottom = floor_bottom - local.y;
                return to_top.max(to_bottom); // Both negative, return closest to surface
            }
        }

        // In the walls (between inner and outer boundary, above floor, below wall height)
        if above_floor && local.y <= self.wall_height {
            let in_wall_x = local.x.abs() >= inner_x && local.x.abs() <= outer_x;
            let in_wall_z = local.z.abs() >= inner_z && local.z.abs() <= outer_z;

            if (in_wall_x && local.z.abs() <= outer_z) || (in_wall_z && local.x.abs() <= outer_x) {
                // Inside wall solid - return negative distance
                let dx = if local.x.abs() > inner_x {
                    inner_x - local.x.abs() // Negative
                } else {
                    local.x.abs() - inner_x // Would be negative if we're past inner
                };
                let dz = if local.z.abs() > inner_z {
                    inner_z - local.z.abs()
                } else {
                    local.z.abs() - inner_z
                };
                return dx.max(dz); // Return the least negative (closest to inner surface)
            }
        }

        // Outside the box entirely - return positive distance to nearest surface
        // This is a simplified exterior SDF
        let dx = (local.x.abs() - outer_x).max(0.0);
        let dz = (local.z.abs() - outer_z).max(0.0);
        let dy = if local.y < floor_bottom {
            floor_bottom - local.y
        } else {
            0.0
        };

        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Get the Y coordinate of the floor surface
    pub fn floor_y(&self) -> f32 {
        self.center.y
    }
}

/// Test ramp - thick angled slab for friction/sliding tests
/// The ramp is a solid slab with configurable thickness
#[derive(Debug, Clone)]
pub struct TestRamp {
    pub base: Vec3,     // Bottom corner of ramp (where it meets floor)
    pub length: f32,    // Length along slope
    pub width: f32,     // Width perpendicular to slope direction
    pub angle_deg: f32, // Angle from horizontal (0-90)
    pub thickness: f32, // Thickness of the ramp slab
}

impl TestRamp {
    pub fn new(base: Vec3, length: f32, width: f32, angle_deg: f32) -> Self {
        Self {
            base,
            length,
            width,
            angle_deg: angle_deg.clamp(0.0, 90.0),
            thickness: 0.1, // 10cm thick
        }
    }

    pub fn with_thickness(
        base: Vec3,
        length: f32,
        width: f32,
        angle_deg: f32,
        thickness: f32,
    ) -> Self {
        Self {
            base,
            length,
            width,
            angle_deg: angle_deg.clamp(0.0, 90.0),
            thickness,
        }
    }

    /// SDF value at a point
    pub fn sdf(&self, point: Vec3) -> f32 {
        let local = point - self.base;

        // Ramp runs along +X direction, tilted up
        let angle_rad = self.angle_deg.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        // Position along the ramp direction (horizontal distance)
        let along_ramp = local.x * cos_a + local.y * sin_a;

        // Check width bounds
        let across_ramp = local.z;
        if across_ramp.abs() > self.width / 2.0 {
            return f32::MAX; // Outside width
        }

        // Check length bounds
        if along_ramp < 0.0 || along_ramp > self.length {
            return f32::MAX; // Outside length
        }

        // Distance perpendicular to ramp surface (positive = above, negative = inside)
        // Normal vector points "up" from ramp: (-sin(θ), cos(θ), 0)
        let dist_to_top = -local.x * sin_a + local.y * cos_a;
        let dist_to_bottom = dist_to_top + self.thickness;

        if dist_to_top > 0.0 {
            // Above ramp surface
            dist_to_top
        } else if dist_to_bottom < 0.0 {
            // Below ramp (through thickness) - shouldn't normally happen
            -dist_to_bottom
        } else {
            // Inside ramp slab - negative value
            dist_to_top // Already negative
        }
    }

    /// Get the height at a given X position along the ramp
    pub fn height_at(&self, x: f32) -> f32 {
        let angle_rad = self.angle_deg.to_radians();
        self.base.y + (x - self.base.x) * angle_rad.tan()
    }
}

/// Test chute - angled channel with thick walls and floor for flow tests
/// The chute runs along the +X direction, tilted down by angle_deg
#[derive(Debug, Clone)]
pub struct TestChute {
    pub inlet: Vec3,          // Top of chute (inlet position, center of channel floor)
    pub length: f32,          // Length along slope
    pub width: f32,           // Inner width between walls
    pub wall_height: f32,     // Height of side walls above floor
    pub angle_deg: f32,       // Angle from horizontal (positive = down slope in +X)
    pub wall_thickness: f32,  // Thickness of walls
    pub floor_thickness: f32, // Thickness of floor
}

impl TestChute {
    pub fn new(inlet: Vec3, length: f32, width: f32, wall_height: f32, angle_deg: f32) -> Self {
        Self {
            inlet,
            length,
            width,
            wall_height,
            angle_deg: angle_deg.clamp(0.0, 90.0),
            wall_thickness: 0.1,  // 10cm walls
            floor_thickness: 0.1, // 10cm floor
        }
    }

    pub fn with_thickness(
        inlet: Vec3,
        length: f32,
        width: f32,
        wall_height: f32,
        angle_deg: f32,
        wall_thickness: f32,
        floor_thickness: f32,
    ) -> Self {
        Self {
            inlet,
            length,
            width,
            wall_height,
            angle_deg: angle_deg.clamp(0.0, 90.0),
            wall_thickness,
            floor_thickness,
        }
    }

    /// SDF value at a point
    /// The chute floor is a thick angled slab, with thick walls on either side
    pub fn sdf(&self, point: Vec3) -> f32 {
        let angle_rad = self.angle_deg.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let rel = point - self.inlet;

        // Position along chute (horizontal distance in X)
        let along = rel.x;

        // Check bounds along chute (with some margin for wall thickness)
        let chute_end_x = self.length * cos_a;
        if along < -self.wall_thickness || along > chute_end_x + self.wall_thickness {
            return f32::MAX; // Outside chute length
        }

        // Height above the tilted floor at this X position
        // Floor surface equation: y = inlet.y - x * tan(angle)
        let floor_y_at_x = -along * sin_a / cos_a;
        let height_above_floor = rel.y - floor_y_at_x;

        // Inner wall boundaries
        let half_width = self.width / 2.0;
        let wall_dist_inner = half_width - rel.z.abs();

        // Outer wall boundaries
        let outer_half_width = half_width + self.wall_thickness;
        let in_outer_bounds = rel.z.abs() <= outer_half_width;

        // Inside open channel
        if along >= 0.0
            && along <= chute_end_x
            && wall_dist_inner > 0.0
            && height_above_floor > 0.0
            && height_above_floor < self.wall_height
        {
            // In open channel - positive distance to nearest surface
            return height_above_floor.min(wall_dist_inner);
        }

        // In floor (below floor surface, within floor thickness)
        if height_above_floor <= 0.0 && height_above_floor >= -self.floor_thickness {
            if along >= 0.0 && along <= chute_end_x && in_outer_bounds {
                return height_above_floor; // Negative
            }
        }

        // In walls
        if along >= 0.0
            && along <= chute_end_x
            && height_above_floor > 0.0
            && height_above_floor <= self.wall_height
        {
            if rel.z.abs() >= half_width && rel.z.abs() <= outer_half_width {
                // In wall material
                return wall_dist_inner; // Negative
            }
        }

        // Outside - return positive distance
        f32::MAX
    }

    /// Get the outlet position (bottom of chute, center of channel)
    pub fn outlet(&self) -> Vec3 {
        let angle_rad = self.angle_deg.to_radians();
        Vec3::new(
            self.inlet.x + self.length * angle_rad.cos(),
            self.inlet.y - self.length * angle_rad.sin(),
            self.inlet.z,
        )
    }
}

/// Generate SDF grid from test geometry
pub struct TestSdfGenerator {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub cell_size: f32,
    pub offset: Vec3,
    pub sdf: Vec<f32>,
}

impl TestSdfGenerator {
    /// Create a new SDF generator with given grid dimensions
    pub fn new(width: usize, height: usize, depth: usize, cell_size: f32, offset: Vec3) -> Self {
        let size = width * height * depth;
        Self {
            width,
            height,
            depth,
            cell_size,
            offset,
            sdf: vec![f32::MAX; size],
        }
    }

    /// Cell index from grid coordinates
    pub fn cell_index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.width * self.height + y * self.width + x
    }

    /// World position of grid cell center
    pub fn cell_center(&self, x: usize, y: usize, z: usize) -> Vec3 {
        self.offset
            + Vec3::new(
                (x as f32 + 0.5) * self.cell_size,
                (y as f32 + 0.5) * self.cell_size,
                (z as f32 + 0.5) * self.cell_size,
            )
    }

    /// Add a TestFloor to the SDF grid
    pub fn add_floor(&mut self, floor: &TestFloor) {
        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    let pos = self.cell_center(x, y, z);
                    let dist = floor.sdf(pos);
                    let idx = self.cell_index(x, y, z);
                    self.sdf[idx] = self.sdf[idx].min(dist);
                }
            }
        }
    }

    /// Add a TestBox to the SDF grid
    pub fn add_box(&mut self, test_box: &TestBox) {
        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    let pos = self.cell_center(x, y, z);
                    let dist = test_box.sdf(pos);
                    let idx = self.cell_index(x, y, z);
                    self.sdf[idx] = self.sdf[idx].min(dist);
                }
            }
        }
    }

    /// Add a TestRamp to the SDF grid
    pub fn add_ramp(&mut self, ramp: &TestRamp) {
        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    let pos = self.cell_center(x, y, z);
                    let dist = ramp.sdf(pos);
                    let idx = self.cell_index(x, y, z);
                    if dist < f32::MAX {
                        self.sdf[idx] = self.sdf[idx].min(dist);
                    }
                }
            }
        }
    }

    /// Add a TestChute to the SDF grid
    pub fn add_chute(&mut self, chute: &TestChute) {
        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    let pos = self.cell_center(x, y, z);
                    let dist = chute.sdf(pos);
                    let idx = self.cell_index(x, y, z);
                    if dist < f32::MAX {
                        self.sdf[idx] = self.sdf[idx].min(dist);
                    }
                }
            }
        }
    }

    /// Get SDF parameters for use with DEM simulation
    pub fn sdf_slice(&self) -> &[f32] {
        &self.sdf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_floor_sdf() {
        // Floor at y=0.5, thickness=0.5 (default), so slab is from y=0 to y=0.5
        let floor = TestFloor::new(0.5);

        // Above floor (y=1.0 > floor_top=0.5)
        let above = floor.sdf(Vec3::new(0.0, 1.0, 0.0));
        assert!(above > 0.0, "Above floor should be positive: {}", above);
        assert!(
            (above - 0.5).abs() < 0.001,
            "Distance above should be 0.5: {}",
            above
        );

        // On floor surface (y=0.5 = floor_top)
        let on_surface = floor.sdf(Vec3::new(0.0, 0.5, 0.0));
        assert!(
            on_surface.abs() < 0.001,
            "On floor surface should be ~0: {}",
            on_surface
        );

        // Inside floor slab (y=0.25 is midway between 0 and 0.5)
        let inside = floor.sdf(Vec3::new(0.0, 0.25, 0.0));
        assert!(
            inside < 0.0,
            "Inside floor slab should be negative: {}",
            inside
        );
        assert!(
            (inside + 0.25).abs() < 0.001,
            "Distance inside should be -0.25: {}",
            inside
        );

        // Below floor bottom (y=-0.5 < floor_bottom=0.0)
        let below = floor.sdf(Vec3::new(0.0, -0.5, 0.0));
        assert!(
            below > 0.0,
            "Below floor bottom should be positive: {}",
            below
        );
    }

    #[test]
    fn test_box_sdf() {
        // Box: 2m x 2m, centered at origin, floor at y=0
        // Inner half_width=1.0, wall_thickness=0.1, so outer boundary at x=1.1
        // Floor: floor_thickness=0.2, so floor bottom at y=-0.2
        let test_box = TestBox::new(Vec3::new(0.0, 0.0, 0.0), 2.0, 2.0, 1.0);

        // Inside box center (open area) - should be positive
        let inside = test_box.sdf(Vec3::new(0.0, 0.5, 0.0));
        assert!(
            inside > 0.0,
            "Inside box center should be positive: {}",
            inside
        );

        // Below floor surface (inside floor slab at y=-0.1) - should be negative
        let below = test_box.sdf(Vec3::new(0.0, -0.1, 0.0));
        assert!(
            below < 0.0,
            "Inside floor slab should be negative: {}",
            below
        );

        // Inside wall (x=1.05 is between inner=1.0 and outer=1.1) - should be negative
        let in_wall = test_box.sdf(Vec3::new(1.05, 0.5, 0.0));
        assert!(in_wall < 0.0, "Inside wall should be negative: {}", in_wall);

        // Outside box entirely (x=1.5 > outer=1.1) - should be positive
        let outside_box = test_box.sdf(Vec3::new(1.5, 0.5, 0.0));
        assert!(
            outside_box > 0.0,
            "Outside box should be positive: {}",
            outside_box
        );

        // Near wall but inside - should be positive but small
        let near_wall = test_box.sdf(Vec3::new(0.9, 0.5, 0.0));
        assert!(
            near_wall > 0.0,
            "Near wall inside should be positive: {}",
            near_wall
        );
        assert!(
            near_wall < 0.2,
            "Near wall should be close to wall: {}",
            near_wall
        );
    }

    #[test]
    fn test_ramp_sdf() {
        let ramp = TestRamp::new(Vec3::ZERO, 1.0, 1.0, 45.0);

        // On ramp surface (midpoint)
        let on_ramp = ramp.sdf(Vec3::new(0.5, 0.5, 0.0));
        assert!(
            on_ramp.abs() < 0.1,
            "On ramp should be near zero: {}",
            on_ramp
        );

        // Above ramp
        let above = ramp.sdf(Vec3::new(0.5, 1.0, 0.0));
        assert!(above > 0.0, "Above ramp should be positive: {}", above);
    }

    #[test]
    fn test_chute_sdf() {
        // Chute: inlet at (0,1,0), 2m long, 0.5m wide, 30 degree slope
        // Inner half_width=0.25, wall_thickness=0.1, so outer wall at z=0.35
        // floor_thickness=0.1
        let chute = TestChute::new(Vec3::new(0.0, 1.0, 0.0), 2.0, 0.5, 0.3, 30.0);

        // At inlet, slightly above floor - should be positive
        let at_inlet = chute.sdf(Vec3::new(0.1, 1.05, 0.0));
        assert!(
            at_inlet > 0.0,
            "At inlet above floor should be positive: {}",
            at_inlet
        );

        // Below inlet floor surface (inside floor slab) - should be negative
        // At x=0.1, floor surface is at y ≈ 1.0 - 0.1*tan(30°) ≈ 0.942
        // So y=0.9 is below the floor surface
        let below_inlet = chute.sdf(Vec3::new(0.1, 0.9, 0.0));
        assert!(
            below_inlet < 0.0,
            "Below inlet floor should be negative: {}",
            below_inlet
        );

        // Inside wall (z=0.3 is between inner=0.25 and outer=0.35) - should be negative
        // At x=0.5, floor is at y ≈ 1.0 - 0.5*tan(30°) ≈ 0.71, so y=0.8 is above floor
        let in_wall = chute.sdf(Vec3::new(0.5, 0.8, 0.3));
        assert!(in_wall < 0.0, "Inside wall should be negative: {}", in_wall);

        // Check outlet position is correct
        let outlet = chute.outlet();
        let expected_outlet_x = 2.0 * 30.0_f32.to_radians().cos();
        let expected_outlet_y = 1.0 - 2.0 * 30.0_f32.to_radians().sin();
        assert!(
            (outlet.x - expected_outlet_x).abs() < 0.01,
            "Outlet X: {}",
            outlet.x
        );
        assert!(
            (outlet.y - expected_outlet_y).abs() < 0.01,
            "Outlet Y: {}",
            outlet.y
        );
    }

    #[test]
    fn test_sdf_generator() {
        let mut gen = TestSdfGenerator::new(10, 10, 10, 0.1, Vec3::ZERO);
        let floor = TestFloor::new(0.2);
        gen.add_floor(&floor);

        // Check that cells below floor have negative SDF
        let below_idx = gen.cell_index(5, 1, 5); // y=0.15
        assert!(gen.sdf[below_idx] < 0.0);

        // Check that cells above floor have positive SDF
        let above_idx = gen.cell_index(5, 5, 5); // y=0.55
        assert!(gen.sdf[above_idx] > 0.0);
    }
}
