# Equipment Geometry Module

## Overview

Create a parametric geometry module for common wash plant equipment shapes. Each shape generates:
1. Solid cell positions for simulation grids (voxelized)
2. Indexed mesh for GPU rendering

This complements the existing `sluice_geometry.rs` which handles sluice-specific logic.

## File Location

`crates/game/src/equipment_geometry.rs`

Add to `crates/game/src/lib.rs`:
```rust
pub mod equipment_geometry;
```

## Core Types

### Shared Vertex Type

Reuse `SluiceVertex` from `sluice_geometry.rs`:
```rust
pub use crate::sluice_geometry::SluiceVertex;
```

### Face Flags

```rust
bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Faces: u8 {
        const NONE   = 0b000000;
        const NEG_X  = 0b000001;  // Left
        const POS_X  = 0b000010;  // Right
        const NEG_Y  = 0b000100;  // Bottom
        const POS_Y  = 0b001000;  // Top
        const NEG_Z  = 0b010000;  // Front
        const POS_Z  = 0b100000;  // Back
        const ALL    = 0b111111;
        const TOP    = Self::POS_Y.bits();
        const BOTTOM = Self::NEG_Y.bits();
        const LEFT   = Self::NEG_X.bits();
        const RIGHT  = Self::POS_X.bits();
        const FRONT  = Self::NEG_Z.bits();
        const BACK   = Self::POS_Z.bits();
    }
}
```

### Hole Pattern (for Grates)

```rust
#[derive(Clone, Copy, Debug, Default)]
pub enum HolePattern {
    #[default]
    Grid,           // Regular grid of circular holes
    Staggered,      // Offset rows for better coverage
    SlotsX,         // Rectangular slots along X axis
    SlotsZ,         // Rectangular slots along Z axis
}
```

---

## Shape 1: Grate (Perforated Plate)

Flat plate with regular holes or slots for size classification.

### Config

```rust
#[derive(Clone, Debug)]
pub struct GrateConfig {
    /// Dimensions in cells: [width_x, thickness_y, depth_z]
    pub dims: [usize; 3],
    /// Cell size in world units
    pub cell_size: f32,
    /// Hole radius in cells (for circular holes)
    pub hole_radius: f32,
    /// Spacing between hole centers in cells
    pub hole_spacing: f32,
    /// Hole pattern type
    pub pattern: HolePattern,
    /// Slot dimensions [width, length] in cells (for slot patterns)
    pub slot_dims: [f32; 2],
    /// Margin from edges before holes start (in cells)
    pub edge_margin: usize,
    /// Colors
    pub color_top: [f32; 4],
    pub color_side: [f32; 4],
    pub color_bottom: [f32; 4],
}
```

### Default

```rust
impl Default for GrateConfig {
    fn default() -> Self {
        Self {
            dims: [40, 1, 20],
            cell_size: 0.25,
            hole_radius: 1.5,
            hole_spacing: 4.0,
            pattern: HolePattern::Grid,
            slot_dims: [1.0, 3.0],
            edge_margin: 2,
            color_top: [0.5, 0.5, 0.55, 1.0],
            color_side: [0.4, 0.4, 0.45, 1.0],
            color_bottom: [0.3, 0.3, 0.35, 1.0],
        }
    }
}
```

### Hole Detection Logic

For a cell at `(i, j, k)` in a grate starting at origin:
- **Grid pattern**: Check distance from nearest hole center
- **Staggered pattern**: Offset every other row by `hole_spacing / 2`
- **Slots**: Check if cell falls within slot rectangle

```rust
impl GrateConfig {
    /// Returns true if cell (i, j, k) is solid (not a hole)
    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        // Outside grate bounds
        if i >= self.dims[0] || j >= self.dims[1] || k >= self.dims[2] {
            return false;
        }

        // Edge margin is always solid
        if i < self.edge_margin || i >= self.dims[0] - self.edge_margin ||
           k < self.edge_margin || k >= self.dims[2] - self.edge_margin {
            return true;
        }

        // Check if in a hole based on pattern
        !self.is_in_hole(i, k)
    }

    fn is_in_hole(&self, i: usize, k: usize) -> bool {
        match self.pattern {
            HolePattern::Grid => self.is_in_grid_hole(i, k),
            HolePattern::Staggered => self.is_in_staggered_hole(i, k),
            HolePattern::SlotsX => self.is_in_slot_x(i, k),
            HolePattern::SlotsZ => self.is_in_slot_z(i, k),
        }
    }
}
```

### Acceptance Criteria

- [ ] Grate with no holes (hole_radius=0) produces solid rectangular plate
- [ ] Grid pattern produces evenly spaced circular holes
- [ ] Staggered pattern offsets alternate rows
- [ ] Slot patterns produce rectangular openings
- [ ] Edge margin creates solid border around holes
- [ ] Mesh only renders exposed faces (not internal hole surfaces facing solid)

### Tests

```rust
#[test]
fn test_grate_solid_plate() {
    let config = GrateConfig {
        dims: [10, 1, 10],
        hole_radius: 0.0, // No holes
        ..Default::default()
    };
    let builder = GrateBuilder::new(config);
    let solid_count = builder.solid_cells().count();
    assert_eq!(solid_count, 10 * 1 * 10); // All cells solid
}

#[test]
fn test_grate_has_holes() {
    let config = GrateConfig {
        dims: [20, 1, 20],
        hole_radius: 2.0,
        hole_spacing: 5.0,
        edge_margin: 2,
        ..Default::default()
    };
    let builder = GrateBuilder::new(config);
    let solid_count = builder.solid_cells().count();
    let total_cells = 20 * 1 * 20;
    assert!(solid_count < total_cells, "Should have holes");
    assert!(solid_count > total_cells / 2, "Should be mostly solid");
}

#[test]
fn test_grate_edge_margin_solid() {
    let config = GrateConfig {
        dims: [20, 1, 20],
        hole_radius: 2.0,
        edge_margin: 3,
        ..Default::default()
    };
    // All cells in edge margin should be solid
    for i in 0..3 {
        for k in 0..20 {
            assert!(config.is_solid(i, 0, k), "Left edge should be solid");
        }
    }
}
```

---

## Shape 2: Box (Open Container)

Rectangular container with configurable open faces.

### Config

```rust
#[derive(Clone, Debug)]
pub struct BoxConfig {
    /// Inner dimensions in cells: [width_x, height_y, depth_z]
    pub inner_dims: [usize; 3],
    /// Wall thickness in cells
    pub wall_thickness: usize,
    /// Cell size in world units
    pub cell_size: f32,
    /// Which faces are open (not solid)
    pub open_faces: Faces,
    /// Colors
    pub color_top: [f32; 4],
    pub color_side: [f32; 4],
    pub color_bottom: [f32; 4],
}
```

### Default

```rust
impl Default for BoxConfig {
    fn default() -> Self {
        Self {
            inner_dims: [20, 10, 20],
            wall_thickness: 2,
            cell_size: 0.25,
            open_faces: Faces::TOP, // Open-top box
            color_top: [0.45, 0.35, 0.25, 1.0],
            color_side: [0.35, 0.28, 0.20, 1.0],
            color_bottom: [0.25, 0.20, 0.15, 1.0],
        }
    }
}
```

### Solid Detection

```rust
impl BoxConfig {
    /// Outer dimensions including walls
    pub fn outer_dims(&self) -> [usize; 3] {
        [
            self.inner_dims[0] + 2 * self.wall_thickness,
            self.inner_dims[1] + 2 * self.wall_thickness,
            self.inner_dims[2] + 2 * self.wall_thickness,
        ]
    }

    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        let outer = self.outer_dims();
        let t = self.wall_thickness;

        // Outside bounds
        if i >= outer[0] || j >= outer[1] || k >= outer[2] {
            return false;
        }

        // Inside interior (hollow)
        let in_interior = i >= t && i < outer[0] - t &&
                          j >= t && j < outer[1] - t &&
                          k >= t && k < outer[2] - t;
        if in_interior {
            return false;
        }

        // Check if on an open face
        if self.open_faces.contains(Faces::NEG_X) && i < t { return false; }
        if self.open_faces.contains(Faces::POS_X) && i >= outer[0] - t { return false; }
        if self.open_faces.contains(Faces::NEG_Y) && j < t { return false; }
        if self.open_faces.contains(Faces::POS_Y) && j >= outer[1] - t { return false; }
        if self.open_faces.contains(Faces::NEG_Z) && k < t { return false; }
        if self.open_faces.contains(Faces::POS_Z) && k >= outer[2] - t { return false; }

        true
    }
}
```

### Acceptance Criteria

- [ ] Closed box (no open faces) produces hollow rectangular prism
- [ ] Open-top box has no cells in top wall region
- [ ] Multiple open faces work together (e.g., TOP | FRONT)
- [ ] Wall thickness is respected on all sides
- [ ] Interior is hollow

### Tests

```rust
#[test]
fn test_box_closed() {
    let config = BoxConfig {
        inner_dims: [4, 4, 4],
        wall_thickness: 1,
        open_faces: Faces::NONE,
        ..Default::default()
    };
    let builder = BoxBuilder::new(config);
    // Outer: 6x6x6 = 216, Inner: 4x4x4 = 64, Solid = 216 - 64 = 152
    assert_eq!(builder.solid_cells().count(), 152);
}

#[test]
fn test_box_open_top() {
    let config = BoxConfig {
        inner_dims: [4, 4, 4],
        wall_thickness: 1,
        open_faces: Faces::TOP,
        ..Default::default()
    };
    let builder = BoxBuilder::new(config);
    // Top wall removed: 152 - (6*6) = 152 - 36 = 116
    assert_eq!(builder.solid_cells().count(), 116);
}

#[test]
fn test_box_interior_empty() {
    let config = BoxConfig {
        inner_dims: [4, 4, 4],
        wall_thickness: 1,
        open_faces: Faces::NONE,
        ..Default::default()
    };
    // Center cell should not be solid
    assert!(!config.is_solid(3, 3, 3));
}
```

---

## Shape 3: Gutter (U-Channel)

U-shaped channel, open on top and both ends (±X).

### Config

```rust
#[derive(Clone, Debug)]
pub struct GutterConfig {
    /// Length along X axis in cells
    pub length: usize,
    /// Inner width (Z) in cells
    pub inner_width: usize,
    /// Inner height (Y) in cells
    pub inner_height: usize,
    /// Wall/floor thickness in cells
    pub wall_thickness: usize,
    /// Cell size in world units
    pub cell_size: f32,
    /// Slope: height difference from start to end in cells (positive = downstream lower)
    pub slope_drop: i32,
    /// Colors
    pub color_top: [f32; 4],
    pub color_side: [f32; 4],
    pub color_bottom: [f32; 4],
}
```

### Default

```rust
impl Default for GutterConfig {
    fn default() -> Self {
        Self {
            length: 40,
            inner_width: 8,
            inner_height: 6,
            wall_thickness: 2,
            cell_size: 0.25,
            slope_drop: 0,
            color_top: [0.5, 0.45, 0.4, 1.0],
            color_side: [0.4, 0.36, 0.32, 1.0],
            color_bottom: [0.3, 0.27, 0.24, 1.0],
        }
    }
}
```

### Solid Detection

```rust
impl GutterConfig {
    pub fn outer_dims(&self) -> [usize; 3] {
        [
            self.length,
            self.inner_height + self.wall_thickness + self.slope_drop.unsigned_abs() as usize,
            self.inner_width + 2 * self.wall_thickness,
        ]
    }

    pub fn floor_height_at(&self, x: usize) -> usize {
        let t = x as f32 / self.length.max(1) as f32;
        let base = self.wall_thickness as f32;
        let drop = self.slope_drop as f32 * t;
        (base - drop).max(0.0) as usize
    }

    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        let t = self.wall_thickness;
        let outer = self.outer_dims();

        if i >= outer[0] || j >= outer[1] || k >= outer[2] {
            return false;
        }

        let floor_j = self.floor_height_at(i);

        // Floor
        if j <= floor_j {
            return true;
        }

        // Side walls (not open ends)
        let wall_top = floor_j + self.inner_height;
        if (k < t || k >= outer[2] - t) && j <= wall_top {
            return true;
        }

        false
    }
}
```

### Acceptance Criteria

- [ ] Flat gutter (slope_drop=0) produces U-channel
- [ ] Sloped gutter has floor height varying along X
- [ ] Open on both X ends
- [ ] Open on top
- [ ] Side walls extend to correct height

### Tests

```rust
#[test]
fn test_gutter_flat() {
    let config = GutterConfig {
        length: 10,
        inner_width: 4,
        inner_height: 3,
        wall_thickness: 1,
        slope_drop: 0,
        ..Default::default()
    };
    // Floor height should be constant
    assert_eq!(config.floor_height_at(0), 1);
    assert_eq!(config.floor_height_at(5), 1);
    assert_eq!(config.floor_height_at(9), 1);
}

#[test]
fn test_gutter_sloped() {
    let config = GutterConfig {
        length: 10,
        inner_width: 4,
        inner_height: 3,
        wall_thickness: 1,
        slope_drop: 2,
        ..Default::default()
    };
    assert!(config.floor_height_at(0) > config.floor_height_at(9));
}

#[test]
fn test_gutter_open_ends() {
    let config = GutterConfig::default();
    let builder = GutterBuilder::new(config.clone());
    // No cells at x=0 should block the opening (except floor/walls)
    // The center of the channel at x=0 should be open
    let center_k = config.inner_width / 2 + config.wall_thickness;
    let mid_j = config.wall_thickness + config.inner_height / 2;
    assert!(!config.is_solid(0, mid_j, center_k), "Channel should be open at ends");
}
```

---

## Shape 4: Hopper (Tapered Funnel)

Funnel shape for feeding material, open on top, with smaller exit at bottom.

### Config

```rust
#[derive(Clone, Debug)]
pub struct HopperConfig {
    /// Top opening dimensions [width_x, depth_z] in cells
    pub top_dims: [usize; 2],
    /// Bottom opening dimensions [width_x, depth_z] in cells
    pub bottom_dims: [usize; 2],
    /// Total height in cells
    pub height: usize,
    /// Wall thickness in cells
    pub wall_thickness: usize,
    /// Cell size in world units
    pub cell_size: f32,
    /// Colors
    pub color_top: [f32; 4],
    pub color_side: [f32; 4],
    pub color_bottom: [f32; 4],
}
```

### Default

```rust
impl Default for HopperConfig {
    fn default() -> Self {
        Self {
            top_dims: [24, 24],
            bottom_dims: [8, 8],
            height: 16,
            wall_thickness: 2,
            cell_size: 0.25,
            color_top: [0.55, 0.5, 0.45, 1.0],
            color_side: [0.45, 0.4, 0.35, 1.0],
            color_bottom: [0.35, 0.3, 0.25, 1.0],
        }
    }
}
```

### Solid Detection

The hopper walls taper linearly from top to bottom. At each height `j`, calculate the opening dimensions by linear interpolation.

```rust
impl HopperConfig {
    pub fn outer_dims(&self) -> [usize; 3] {
        [
            self.top_dims[0] + 2 * self.wall_thickness,
            self.height,
            self.top_dims[1] + 2 * self.wall_thickness,
        ]
    }

    /// Get inner opening bounds at height j: (min_x, max_x, min_z, max_z)
    pub fn opening_at(&self, j: usize) -> (usize, usize, usize, usize) {
        let t = j as f32 / self.height.max(1) as f32; // 0 at bottom, 1 at top
        let outer = self.outer_dims();

        let inner_x = lerp(self.bottom_dims[0] as f32, self.top_dims[0] as f32, t) as usize;
        let inner_z = lerp(self.bottom_dims[1] as f32, self.top_dims[1] as f32, t) as usize;

        let margin_x = (outer[0] - inner_x) / 2;
        let margin_z = (outer[2] - inner_z) / 2;

        (margin_x, outer[0] - margin_x, margin_z, outer[2] - margin_z)
    }

    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        let outer = self.outer_dims();

        if i >= outer[0] || j >= outer[1] || k >= outer[2] {
            return false;
        }

        let (min_x, max_x, min_z, max_z) = self.opening_at(j);

        // Inside the opening = not solid
        if i >= min_x && i < max_x && k >= min_z && k < max_z {
            return false;
        }

        // On the walls = solid
        true
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
```

### Acceptance Criteria

- [ ] Top opening matches top_dims
- [ ] Bottom opening matches bottom_dims
- [ ] Walls taper linearly between top and bottom
- [ ] Open on top and bottom (funnel through)
- [ ] No floor (material falls through)

### Tests

```rust
#[test]
fn test_hopper_top_opening() {
    let config = HopperConfig {
        top_dims: [20, 20],
        bottom_dims: [4, 4],
        height: 10,
        wall_thickness: 2,
        ..Default::default()
    };
    let (min_x, max_x, min_z, max_z) = config.opening_at(9); // Near top
    assert_eq!(max_x - min_x, 20, "Top opening width");
    assert_eq!(max_z - min_z, 20, "Top opening depth");
}

#[test]
fn test_hopper_bottom_opening() {
    let config = HopperConfig {
        top_dims: [20, 20],
        bottom_dims: [4, 4],
        height: 10,
        wall_thickness: 2,
        ..Default::default()
    };
    let (min_x, max_x, min_z, max_z) = config.opening_at(0); // Bottom
    assert_eq!(max_x - min_x, 4, "Bottom opening width");
}

#[test]
fn test_hopper_center_open() {
    let config = HopperConfig {
        top_dims: [20, 20],
        bottom_dims: [4, 4],
        height: 10,
        wall_thickness: 2,
        ..Default::default()
    };
    let outer = config.outer_dims();
    let center_x = outer[0] / 2;
    let center_z = outer[2] / 2;
    // Center should be open at all heights
    for j in 0..config.height {
        assert!(!config.is_solid(center_x, j, center_z), "Center should be open at j={}", j);
    }
}
```

---

## Shape 5: Chute (Angled Slide)

Angled slide with side walls for transferring material between stages.

### Config

```rust
#[derive(Clone, Debug)]
pub struct ChuteConfig {
    /// Length along X axis in cells
    pub length: usize,
    /// Inner width (Z) in cells
    pub inner_width: usize,
    /// Floor height at start (X=0) in cells
    pub start_height: usize,
    /// Floor height at end (X=length-1) in cells
    pub end_height: usize,
    /// Wall height above floor in cells
    pub wall_height: usize,
    /// Wall/floor thickness in cells
    pub thickness: usize,
    /// Cell size in world units
    pub cell_size: f32,
    /// Colors
    pub color_top: [f32; 4],
    pub color_side: [f32; 4],
    pub color_bottom: [f32; 4],
}
```

### Default

```rust
impl Default for ChuteConfig {
    fn default() -> Self {
        Self {
            length: 20,
            inner_width: 10,
            start_height: 8,
            end_height: 2,
            wall_height: 4,
            thickness: 2,
            cell_size: 0.25,
            color_top: [0.5, 0.45, 0.4, 1.0],
            color_side: [0.4, 0.36, 0.32, 1.0],
            color_bottom: [0.3, 0.27, 0.24, 1.0],
        }
    }
}
```

### Solid Detection

```rust
impl ChuteConfig {
    pub fn outer_dims(&self) -> [usize; 3] {
        [
            self.length,
            self.start_height.max(self.end_height) + self.wall_height + self.thickness,
            self.inner_width + 2 * self.thickness,
        ]
    }

    pub fn floor_height_at(&self, x: usize) -> usize {
        let t = x as f32 / self.length.max(1) as f32;
        lerp(self.start_height as f32, self.end_height as f32, t) as usize
    }

    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        let t = self.thickness;
        let outer = self.outer_dims();

        if i >= outer[0] || j >= outer[1] || k >= outer[2] {
            return false;
        }

        let floor_j = self.floor_height_at(i);
        let wall_top = floor_j + self.wall_height;

        // Floor (including thickness below)
        if j >= floor_j && j < floor_j + t {
            return true;
        }

        // Side walls
        if (k < t || k >= outer[2] - t) && j >= floor_j && j <= wall_top {
            return true;
        }

        false
    }
}
```

### Acceptance Criteria

- [ ] Floor slopes from start_height to end_height
- [ ] Open at both X ends
- [ ] Open on top
- [ ] Side walls follow floor slope
- [ ] Wall height is consistent above floor

### Tests

```rust
#[test]
fn test_chute_slope() {
    let config = ChuteConfig {
        length: 10,
        start_height: 8,
        end_height: 2,
        ..Default::default()
    };
    assert_eq!(config.floor_height_at(0), 8);
    assert_eq!(config.floor_height_at(9), 2);
    assert!(config.floor_height_at(5) > 2 && config.floor_height_at(5) < 8);
}

#[test]
fn test_chute_open_ends() {
    let config = ChuteConfig::default();
    let center_k = config.thickness + config.inner_width / 2;
    let mid_j = config.floor_height_at(0) + config.wall_height / 2;
    // Should be open in the channel
    assert!(!config.is_solid(0, mid_j, center_k));
}
```

---

## Shape 6: Frame (Structural Box Frame)

Open box frame - just edges, no faces. For structural supports.

### Config

```rust
#[derive(Clone, Debug)]
pub struct FrameConfig {
    /// Outer dimensions [width_x, height_y, depth_z] in cells
    pub outer_dims: [usize; 3],
    /// Beam thickness in cells
    pub beam_thickness: usize,
    /// Cell size in world units
    pub cell_size: f32,
    /// Color
    pub color: [f32; 4],
}
```

### Default

```rust
impl Default for FrameConfig {
    fn default() -> Self {
        Self {
            outer_dims: [20, 15, 20],
            beam_thickness: 2,
            cell_size: 0.25,
            color: [0.3, 0.3, 0.35, 1.0],
        }
    }
}
```

### Solid Detection

A cell is solid if it's on an edge (within beam_thickness of at least 2 faces).

```rust
impl FrameConfig {
    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        let t = self.beam_thickness;
        let [w, h, d] = self.outer_dims;

        if i >= w || j >= h || k >= d {
            return false;
        }

        // Count how many axes we're near an edge on
        let near_x = i < t || i >= w - t;
        let near_y = j < t || j >= h - t;
        let near_z = k < t || k >= d - t;

        // On an edge = near boundary on at least 2 axes
        (near_x as u8 + near_y as u8 + near_z as u8) >= 2
    }
}
```

### Acceptance Criteria

- [ ] Only edges are solid, faces are open
- [ ] 12 edges of a rectangular prism
- [ ] Beam thickness is respected

### Tests

```rust
#[test]
fn test_frame_center_empty() {
    let config = FrameConfig {
        outer_dims: [10, 10, 10],
        beam_thickness: 2,
        ..Default::default()
    };
    // Center should be empty
    assert!(!config.is_solid(5, 5, 5));
}

#[test]
fn test_frame_corner_solid() {
    let config = FrameConfig {
        outer_dims: [10, 10, 10],
        beam_thickness: 2,
        ..Default::default()
    };
    // Corner should be solid
    assert!(config.is_solid(0, 0, 0));
    assert!(config.is_solid(1, 1, 1));
}

#[test]
fn test_frame_face_center_empty() {
    let config = FrameConfig {
        outer_dims: [10, 10, 10],
        beam_thickness: 2,
        ..Default::default()
    };
    // Center of a face (not an edge) should be empty
    assert!(!config.is_solid(5, 5, 0)); // Front face center
}
```

---

## Shape 7: Baffle (Angled Deflector)

Simple angled plate for redirecting flow.

### Config

```rust
#[derive(Clone, Debug)]
pub struct BaffleConfig {
    /// Width (X) in cells
    pub width: usize,
    /// Height (Y) in cells
    pub height: usize,
    /// Thickness in cells
    pub thickness: usize,
    /// Angle in degrees (0 = vertical, 90 = horizontal)
    pub angle_degrees: f32,
    /// Cell size in world units
    pub cell_size: f32,
    /// Color
    pub color: [f32; 4],
}
```

### Default

```rust
impl Default for BaffleConfig {
    fn default() -> Self {
        Self {
            width: 10,
            height: 8,
            thickness: 2,
            angle_degrees: 45.0,
            cell_size: 0.25,
            color: [0.4, 0.4, 0.45, 1.0],
        }
    }
}
```

### Solid Detection

Angled plate: at each height j, the Z position shifts based on angle.

```rust
impl BaffleConfig {
    pub fn depth(&self) -> usize {
        // Z extent depends on angle
        let shift = (self.height as f32 * self.angle_degrees.to_radians().tan()).abs() as usize;
        self.thickness + shift
    }

    pub fn is_solid(&self, i: usize, j: usize, k: usize) -> bool {
        if i >= self.width || j >= self.height || k >= self.depth() {
            return false;
        }

        // Z position of plate at this height
        let z_offset = (j as f32 * self.angle_degrees.to_radians().tan()) as usize;
        k >= z_offset && k < z_offset + self.thickness
    }
}
```

### Acceptance Criteria

- [ ] Vertical baffle (0°) produces flat plate in XY plane
- [ ] Angled baffle shifts in Z as Y increases
- [ ] Thickness is consistent along the plate

### Tests

```rust
#[test]
fn test_baffle_vertical() {
    let config = BaffleConfig {
        angle_degrees: 0.0,
        thickness: 2,
        ..Default::default()
    };
    // At all heights, plate should be at k=0..2
    assert!(config.is_solid(5, 0, 0));
    assert!(config.is_solid(5, 4, 0));
    assert!(config.is_solid(5, 4, 1));
    assert!(!config.is_solid(5, 4, 2));
}

#[test]
fn test_baffle_angled() {
    let config = BaffleConfig {
        angle_degrees: 45.0,
        height: 8,
        thickness: 2,
        ..Default::default()
    };
    // At j=4, plate should be shifted by ~4 cells in Z
    // Check that k=0 is NOT solid at j=4
    assert!(!config.is_solid(5, 4, 0));
}
```

---

## Builder Pattern

Each shape has a builder that follows the same pattern as `SluiceGeometryBuilder`:

```rust
pub struct GrateBuilder {
    config: GrateConfig,
    vertices: Vec<SluiceVertex>,
    indices: Vec<u32>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
}

impl GrateBuilder {
    pub fn new(config: GrateConfig) -> Self { ... }
    pub fn config(&self) -> &GrateConfig { ... }
    pub fn solid_cells(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ { ... }
    pub fn build_mesh(&mut self) { ... }
    pub fn upload(&mut self, device: &wgpu::Device) { ... }
    pub fn vertex_buffer(&self) -> Option<&wgpu::Buffer> { ... }
    pub fn index_buffer(&self) -> Option<&wgpu::Buffer> { ... }
    pub fn num_indices(&self) -> u32 { ... }
}
```

Consider a macro or trait to reduce duplication:

```rust
pub trait EquipmentBuilder {
    type Config;
    fn new(config: Self::Config) -> Self;
    fn config(&self) -> &Self::Config;
    fn solid_cells(&self) -> Box<dyn Iterator<Item = (usize, usize, usize)> + '_>;
    fn build_mesh(&mut self);
    fn upload(&mut self, device: &wgpu::Device);
    // ... etc
}
```

---

## Integration Test

Create an example that renders multiple equipment pieces together:

```rust
// examples/equipment_gallery.rs
fn main() {
    // Create one of each shape
    let grate = GrateBuilder::new(GrateConfig::default());
    let box_ = BoxBuilder::new(BoxConfig::default());
    let gutter = GutterBuilder::new(GutterConfig::default());
    let hopper = HopperBuilder::new(HopperConfig::default());
    let chute = ChuteBuilder::new(ChuteConfig::default());
    let frame = FrameBuilder::new(FrameConfig::default());
    let baffle = BaffleBuilder::new(BaffleConfig::default());

    // Position them in a gallery layout
    // Render all together
}
```

---

## Acceptance Criteria Summary

### Must Have
- [ ] All 7 shapes implemented with Config + Builder
- [ ] Each shape produces correct solid_cells iterator
- [ ] Each shape builds renderable mesh (vertices + indices)
- [ ] All unit tests pass
- [ ] Reuses SluiceVertex from sluice_geometry.rs
- [ ] `cargo test -p game` passes
- [ ] `cargo check` compiles

### Should Have
- [ ] equipment_gallery example showing all shapes
- [ ] EquipmentBuilder trait to reduce boilerplate

### Nice to Have
- [ ] Composite shapes (e.g., hopper feeding into chute)
- [ ] Transform support (position, rotation)

---

## Dependencies

Add to `crates/game/Cargo.toml` if not present:
```toml
bitflags = "2"
```

---

## File Structure

```
crates/game/src/
├── lib.rs                    # Add: pub mod equipment_geometry;
├── sluice_geometry.rs        # Existing (reuse SluiceVertex)
├── equipment_geometry.rs     # NEW - all shapes in one file
└── ...

examples/
├── equipment_gallery.rs      # NEW - visual test of all shapes
└── ...
```
