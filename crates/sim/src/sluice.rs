//! Sluice box geometry setup
//!
//! Creates the terrain for testing vortex formation:
//! - Sloped floor
//! - Multiple historically accurate riffle geometry options
//!
//! Riffle modes implement different gold recovery strategies:
//! - ClassicBattEdge: Sharp upstream lip, gentle downstream ramp
//! - DoublePocket: Toe + main pocket for nested vortices
//! - ParallelBoards: Longitudinal strips for fines retention
//! - VNotch: Downstream-pointing V for staged flow splitting
//! - StepCascade: Vertical drops with flat ledges for deep settling

use crate::flip::FlipSimulation;

/// Riffle geometry mode selection
/// Each mode creates different vortex and trapping behavior
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum RiffleMode {
    /// No riffles - smooth floor for baseline comparison
    None,
    /// Classic batt edge: sharp upstream lip, gentle downstream ramp
    /// Purpose: baseline vortex + gold trapping
    #[default]
    ClassicBattEdge,
    /// Double pocket: small toe before main lip creating two recirculation zones
    /// Purpose: fine gold + sand trapping, deeper recirculation
    DoublePocket,
    /// Parallel boards: vertical strips aligned with flow
    /// Purpose: fines retention at high flow, multiple small shear layers
    ParallelBoards,
    /// V-notch: downstream-pointing V shapes
    /// Purpose: staged flow splitting + localized pockets
    VNotch,
    /// Step cascade: vertical drop + flat ledge + second drop
    /// Purpose: gentle flow + deep settling, layered recirculation
    StepCascade,
}

impl RiffleMode {
    /// Get the next mode in cycle order
    pub fn next(self) -> Self {
        match self {
            Self::None => Self::ClassicBattEdge,
            Self::ClassicBattEdge => Self::DoublePocket,
            Self::DoublePocket => Self::ParallelBoards,
            Self::ParallelBoards => Self::VNotch,
            Self::VNotch => Self::StepCascade,
            Self::StepCascade => Self::None,
        }
    }

    /// Get display name for UI
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::ClassicBattEdge => "Classic Batt Edge",
            Self::DoublePocket => "Double Pocket",
            Self::ParallelBoards => "Parallel Boards",
            Self::VNotch => "V-Notch",
            Self::StepCascade => "Step Cascade",
        }
    }

    /// Get color for debug overlay (RGBA)
    pub fn debug_color(&self) -> [u8; 4] {
        match self {
            Self::None => [100, 100, 100, 255],          // Gray
            Self::ClassicBattEdge => [200, 100, 50, 255], // Orange
            Self::DoublePocket => [50, 150, 200, 255],    // Cyan
            Self::ParallelBoards => [150, 200, 50, 255],  // Lime
            Self::VNotch => [200, 50, 150, 255],          // Magenta
            Self::StepCascade => [100, 50, 200, 255],     // Purple
        }
    }
}

/// Configuration for sluice geometry
#[derive(Clone, Debug)]
pub struct SluiceConfig {
    /// Floor slope (rise per cell, 0.25 = ~14 degrees)
    pub slope: f32,
    /// Spacing between riffles in grid cells
    pub riffle_spacing: usize,
    /// Height of riffles in grid cells
    pub riffle_height: usize,
    /// Width/depth of riffles in grid cells
    pub riffle_width: usize,
    /// Riffle geometry mode
    pub riffle_mode: RiffleMode,
    /// Length of flat inlet section (slick plate)
    pub slick_plate_len: usize,
}

impl Default for SluiceConfig {
    fn default() -> Self {
        Self {
            slope: 0.25,
            riffle_spacing: 60,
            riffle_height: 6,
            riffle_width: 4,
            riffle_mode: RiffleMode::ClassicBattEdge,
            slick_plate_len: 50,
        }
    }
}

/// Create a sluice box with configurable riffle mode
pub fn create_sluice_with_mode(sim: &mut FlipSimulation, config: &SluiceConfig) {
    let width = sim.grid.width;
    let height = sim.grid.height;
    // dx could be used for dimension scaling if needed
    let _dx = sim.grid.cell_size as usize;

    // Clear all solid
    for idx in 0..sim.grid.solid.len() {
        sim.grid.solid[idx] = false;
    }

    // Create sloped floor
    // Water flows LEFT to RIGHT (downhill)
    let base_height = height / 4;

    for i in 0..width {
        // Floor surface calculation
        let floor_y = if i < config.slick_plate_len {
            // Slick Plate: Flat section
            base_height
        } else {
            // Sloped section
            base_height + ((i - config.slick_plate_len) as f32 * config.slope) as usize
        };

        // Fill everything below floor_y as solid
        for j in floor_y..height {
            sim.grid.set_solid(i, j);
        }

        // Add riffles based on mode
        if config.riffle_mode != RiffleMode::None {
            apply_riffle_geometry(
                sim,
                i,
                floor_y,
                config,
            );
        }
    }

    // Add walls on left and right
    for j in 0..height {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(width - 1, j);
    }

    // Precompute SDF with new geometry
    sim.grid.compute_sdf();
}

/// Apply riffle geometry at a given x position based on mode
fn apply_riffle_geometry(
    sim: &mut FlipSimulation,
    x: usize,
    floor_y: usize,
    config: &SluiceConfig,
) {
    let width = sim.grid.width;
    let riffle_start = config.slick_plate_len + config.riffle_spacing;

    // Check if we're past the inlet and before the outlet
    if x < riffle_start || x >= width - config.riffle_spacing {
        return;
    }

    match config.riffle_mode {
        RiffleMode::None => {}
        RiffleMode::ClassicBattEdge => {
            apply_classic_batt_edge(sim, x, floor_y, config);
        }
        RiffleMode::DoublePocket => {
            apply_double_pocket(sim, x, floor_y, config);
        }
        RiffleMode::ParallelBoards => {
            apply_parallel_boards(sim, x, floor_y, config);
        }
        RiffleMode::VNotch => {
            apply_v_notch(sim, x, floor_y, config);
        }
        RiffleMode::StepCascade => {
            apply_step_cascade(sim, x, floor_y, config);
        }
    }
}

/// ClassicBattEdge: Sharp upstream lip, gentle downstream ramp
/// Profile: Vertical upstream face, 30-40° downstream slope
/// Dimensions: Height = 1.0-1.5 * dx, Spacing = 6-10 * height
fn apply_classic_batt_edge(
    sim: &mut FlipSimulation,
    x: usize,
    floor_y: usize,
    config: &SluiceConfig,
) {
    let cycle_len = config.riffle_spacing;
    let rel_x = (x - config.slick_plate_len - config.riffle_spacing) % cycle_len;

    // Riffle geometry: gentler ramp (3:1 slope) for better flow adherence
    let ramp_slope_inv = 3; // 3 units run for 1 unit rise
    let ramp_len = config.riffle_height * ramp_slope_inv;
    let total_len = ramp_len + config.riffle_width;

    if rel_x < total_len {
        // Calculate height at this x
        let h = if rel_x < ramp_len {
            // Ramp section: rise 1 unit every 3 cells
            (rel_x / ramp_slope_inv) + 1
        } else {
            // Flat top section (sharp upstream edge)
            config.riffle_height
        };

        // Fill solid from floor up to height h
        for dy in 0..h {
            let riffle_y = floor_y.saturating_sub(dy + 1);
            if riffle_y > 0 {
                sim.grid.set_solid(x, riffle_y);
            }
        }
    }
}

/// DoublePocket: Toe + main pocket for nested vortices
/// Profile: Small toe before main lip, two recirculation zones
fn apply_double_pocket(
    sim: &mut FlipSimulation,
    x: usize,
    floor_y: usize,
    config: &SluiceConfig,
) {
    let cycle_len = config.riffle_spacing;
    let rel_x = (x - config.slick_plate_len - config.riffle_spacing) % cycle_len;

    // Toe dimensions (smaller bump before main riffle)
    let toe_height = (config.riffle_height / 2).max(1);
    let toe_width = config.riffle_width / 2;

    // Main riffle dimensions
    let main_height = config.riffle_height;
    let main_width = config.riffle_width;

    // Gap between toe and main
    let gap = 2;

    // Total pattern: toe + gap + main + pocket
    let toe_start = 0;
    let toe_end = toe_width;
    let main_start = toe_end + gap;
    let main_end = main_start + main_width;

    // Downstream ramp for main pocket (deeper recirculation)
    let ramp_len = main_height * 2;
    let total_len = main_end + ramp_len;

    if rel_x < total_len {
        if rel_x >= toe_start && rel_x < toe_end {
            // Toe section
            for dy in 0..toe_height {
                let riffle_y = floor_y.saturating_sub(dy + 1);
                if riffle_y > 0 {
                    sim.grid.set_solid(x, riffle_y);
                }
            }
        } else if rel_x >= main_start && rel_x < main_end {
            // Main lip (sharp upstream edge)
            for dy in 0..main_height {
                let riffle_y = floor_y.saturating_sub(dy + 1);
                if riffle_y > 0 {
                    sim.grid.set_solid(x, riffle_y);
                }
            }
        } else if rel_x >= main_end && rel_x < main_end + ramp_len {
            // Downstream ramp (creates deeper pocket)
            let ramp_progress = rel_x - main_end;
            let h = main_height.saturating_sub(ramp_progress / 2);
            for dy in 0..h {
                let riffle_y = floor_y.saturating_sub(dy + 1);
                if riffle_y > 0 {
                    sim.grid.set_solid(x, riffle_y);
                }
            }
        }
    }
}

/// ParallelBoards: Longitudinal strips aligned with flow
/// Profile: Vertical strips running parallel to flow direction
/// Purpose: Fines retention at high flow with minimal surface disruption
fn apply_parallel_boards(
    sim: &mut FlipSimulation,
    x: usize,
    floor_y: usize,
    config: &SluiceConfig,
) {
    // Board spacing: 1 cell wide boards with 3-5 cell gaps
    let board_width = 1;
    let gap_width = 4;
    let pattern_width = board_width + gap_width;

    // Check if this x position should have a board
    // Boards run the full length of the sluice after slick plate
    let rel_x = x.saturating_sub(config.slick_plate_len);

    // Only apply boards every pattern_width cells
    if rel_x % pattern_width < board_width {
        // Create vertical board (runs in flow direction)
        for dy in 0..config.riffle_height {
            let board_y = floor_y.saturating_sub(dy + 1);
            if board_y > 0 {
                sim.grid.set_solid(x, board_y);
            }
        }
    }
}

/// VNotch: Downstream-pointing V shapes
/// Profile: V-shape pointing downstream creates staged flow splitting
fn apply_v_notch(
    sim: &mut FlipSimulation,
    x: usize,
    floor_y: usize,
    config: &SluiceConfig,
) {
    let cycle_len = config.riffle_spacing;
    let rel_x = (x - config.slick_plate_len - config.riffle_spacing) % cycle_len;

    // V-notch dimensions
    let v_width = config.riffle_width * 2;
    let v_depth = config.riffle_height;

    if rel_x < v_width {
        // Calculate height for V shape
        // Height increases toward edges, minimum at center
        let center = v_width / 2;
        let dist_from_center = if rel_x < center {
            center - rel_x
        } else {
            rel_x - center
        };

        // Height is proportional to distance from center
        let h = (dist_from_center * v_depth / center.max(1)).min(v_depth);

        if h > 0 {
            for dy in 0..h {
                let riffle_y = floor_y.saturating_sub(dy + 1);
                if riffle_y > 0 {
                    sim.grid.set_solid(x, riffle_y);
                }
            }
        }
    }
}

/// StepCascade: Vertical drops with flat ledges
/// Profile: Drop + ledge + drop creating layered recirculation
fn apply_step_cascade(
    sim: &mut FlipSimulation,
    x: usize,
    floor_y: usize,
    config: &SluiceConfig,
) {
    let cycle_len = config.riffle_spacing;
    let rel_x = (x - config.slick_plate_len - config.riffle_spacing) % cycle_len;

    // Step dimensions
    let step_height = config.riffle_height / 2;
    let ledge_width = config.riffle_width;
    let total_width = ledge_width * 2 + config.riffle_width;

    if rel_x < total_width {
        let h = if rel_x < ledge_width {
            // First step (full height)
            config.riffle_height
        } else if rel_x < ledge_width * 2 {
            // Flat ledge (half height)
            step_height
        } else {
            // Second drop returns to floor
            0
        };

        if h > 0 {
            for dy in 0..h {
                let riffle_y = floor_y.saturating_sub(dy + 1);
                if riffle_y > 0 {
                    sim.grid.set_solid(x, riffle_y);
                }
            }
        }
    }
}

/// Create a sluice box test setup (legacy interface)
/// - slope: how much the floor rises per cell (0.3 = 17 degrees)
/// - riffle_spacing: cells between riffles
/// - riffle_height: how tall the riffles are (cells above floor)
/// - riffle_width: how wide each riffle is (cells)
pub fn create_sluice(sim: &mut FlipSimulation, slope: f32, riffle_spacing: usize, riffle_height: usize, riffle_width: usize) {
    let config = SluiceConfig {
        slope,
        riffle_spacing,
        riffle_height,
        riffle_width,
        riffle_mode: RiffleMode::ClassicBattEdge,
        slick_plate_len: 50,
    };
    create_sluice_with_mode(sim, &config);
}

/// Create a simple flat-bottom test tank with riffles
pub fn create_flat_sluice(sim: &mut FlipSimulation, riffle_spacing: usize, riffle_height: usize) {
    let width = sim.grid.width;
    let height = sim.grid.height;

    // Clear all solid
    for idx in 0..sim.grid.solid.len() {
        sim.grid.solid[idx] = false;
    }

    // Floor at bottom 20%
    let floor_y = height - height / 5;

    for i in 0..width {
        // Floor
        for j in floor_y..height {
            sim.grid.set_solid(i, j);
        }

        // Riffles
        if i > riffle_spacing && i % riffle_spacing == 0 && i < width - riffle_spacing {
            for dy in 0..riffle_height {
                let riffle_y = floor_y.saturating_sub(dy + 1);
                if riffle_y > 0 {
                    sim.grid.set_solid(i, riffle_y);
                }
            }
        }
    }

    // Walls
    for j in 0..height {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(width - 1, j);
    }
}

/// Create a simple box with no riffles for basic fluid testing
pub fn create_box(sim: &mut FlipSimulation) {
    let width = sim.grid.width;
    let height = sim.grid.height;

    // Clear all solid
    for idx in 0..sim.grid.solid.len() {
        sim.grid.solid[idx] = false;
    }

    // Floor
    for i in 0..width {
        sim.grid.set_solid(i, height - 1);
    }

    // Walls
    for j in 0..height {
        sim.grid.set_solid(0, j);
        sim.grid.set_solid(width - 1, j);
    }
}

/// Get riffle cell positions for debug rendering
/// Returns a list of (x, y, is_riffle_top) for overlay rendering
pub fn get_riffle_cells(sim: &FlipSimulation, config: &SluiceConfig) -> Vec<(usize, usize, bool)> {
    let mut cells = Vec::new();
    let width = sim.grid.width;
    let height = sim.grid.height;
    let base_height = height / 4;

    for i in 0..width {
        let floor_y = if i < config.slick_plate_len {
            base_height
        } else {
            base_height + ((i - config.slick_plate_len) as f32 * config.slope) as usize
        };

        // Check cells above floor for riffle geometry
        for j in 0..floor_y {
            if sim.grid.is_solid(i, j) {
                // Check if this is a "top" cell (no solid above)
                let is_top = j == 0 || !sim.grid.is_solid(i, j - 1);
                cells.push((i, j, is_top));
            }
        }
    }

    cells
}

/// Compute surface heightfield for rendering
/// Returns y-coordinate of fluid surface for each x column
pub fn compute_surface_heightfield(sim: &FlipSimulation) -> Vec<f32> {
    let width = sim.grid.width;
    let cell_size = sim.grid.cell_size;

    let mut surface = vec![f32::MAX; width];

    // Find topmost water particle in each column
    for particle in sim.particles.iter() {
        if particle.material == crate::particle::ParticleMaterial::Water {
            let i = (particle.position.x / cell_size) as usize;
            if i < width {
                surface[i] = surface[i].min(particle.position.y);
            }
        }
    }

    // Fill gaps with linear interpolation
    let mut last_valid = 0;
    let mut last_y = f32::MAX;

    for i in 0..width {
        if surface[i] < f32::MAX {
            // Fill gaps between last valid and this
            if last_y < f32::MAX && i > last_valid + 1 {
                let span = (i - last_valid) as f32;
                for j in (last_valid + 1)..i {
                    let t = (j - last_valid) as f32 / span;
                    surface[j] = last_y * (1.0 - t) + surface[i] * t;
                }
            }
            last_valid = i;
            last_y = surface[i];
        }
    }

    // Fill trailing gaps
    for i in (last_valid + 1)..width {
        surface[i] = last_y;
    }

    surface
}
