//! Solid cell marking for equipment geometry.
//!
//! This module contains methods to mark cells as solid for different piece types:
//! - Gutters (with variable width support)
//! - Sluices (with riffles)
//! - Shaker decks (with perforated grate)

use crate::editor::{GutterPiece, ShakerDeckPiece, SluicePiece};
use sim3d::FlipSimulation3D;

use super::MultiGridSim;

impl MultiGridSim {
    /// Mark gutter solid cells using cell-index approach.
    /// Supports variable width gutters (funnel effect) via width_at().
    pub fn mark_gutter_solid_cells(
        sim: &mut FlipSimulation3D,
        gutter: &GutterPiece,
        cell_size: f32,
    ) {
        let width = sim.grid.width;
        let height = sim.grid.height;
        let depth = sim.grid.depth;

        // Gutter position gives center in X/Z and base floor height in Y
        let center_i = (gutter.position.x / cell_size).round() as i32;
        let base_j = (gutter.position.y / cell_size).round() as i32;
        let center_k = (gutter.position.z / cell_size).round() as i32;

        // Channel dimensions in cells
        let half_len_cells = ((gutter.length / 2.0) / cell_size).ceil() as i32;
        // Inlet (start) half width for back wall
        let inlet_half_wid_cells = ((gutter.width / 2.0) / cell_size).ceil() as i32;

        // Floor height drop due to angle (in cells)
        let angle_rad = gutter.angle_deg.to_radians();
        let total_drop = gutter.length * angle_rad.tan();
        let half_drop_cells = ((total_drop / 2.0) / cell_size).round() as i32;

        // Floor heights at inlet (left) and outlet (right)
        let _floor_j_left = base_j + half_drop_cells; // Inlet is higher
        let _floor_j_right = base_j - half_drop_cells; // Outlet is lower

        // Wall parameters
        let wall_height_cells = ((gutter.wall_height / cell_size).ceil() as i32).max(8);
        let wall_thick_cells = 2_i32;

        // Channel bounds in i (length direction)
        let i_start = (center_i - half_len_cells).max(0) as usize;
        let i_end = ((center_i + half_len_cells) as usize).min(width);

        // Outlet width (at end of channel)
        let outlet_half_wid_cells = ((gutter.width_at(1.0) / 2.0) / cell_size).ceil() as i32;

        for i in 0..width {
            // Calculate position along gutter (0.0 = inlet, 1.0 = outlet)
            let i_i = i as i32;
            let t = if i_i <= center_i - half_len_cells {
                0.0
            } else if i_i >= center_i + half_len_cells {
                1.0
            } else {
                ((i_i - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };

            // Variable width at this position (funnel effect)
            let local_width = gutter.width_at(t);
            let half_wid_cells = ((local_width / 2.0) / cell_size).ceil() as i32;

            // Channel bounds in k (width direction) - variable!
            let k_start = (center_k - half_wid_cells).max(0) as usize;
            let k_end = ((center_k + half_wid_cells) as usize).min(depth);

            // Compute floor_j directly from mesh floor position to align with visual
            // Mesh floor Y at this position = gutter.position.y + half_drop - t * total_drop
            // Solid top should be at mesh floor, so floor_j = floor(mesh_floor_Y / cell_size)
            let mesh_floor_y = gutter.position.y + (total_drop / 2.0) - t * total_drop;
            let floor_j = (mesh_floor_y / cell_size).floor() as i32;

            // Look BOTH forward AND backward to cover staircase transitions
            // At floor drops, both adjacent cells need the higher floor_j to prevent diagonal gaps
            let t_next = if i_i + 1 >= center_i + half_len_cells {
                1.0
            } else if i_i + 1 <= center_i - half_len_cells {
                0.0
            } else {
                ((i_i + 1 - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };
            let mesh_floor_y_next = gutter.position.y + (total_drop / 2.0) - t_next * total_drop;
            let floor_j_next = (mesh_floor_y_next / cell_size).floor() as i32;

            let t_prev = if i_i - 1 <= center_i - half_len_cells {
                0.0
            } else if i_i - 1 >= center_i + half_len_cells {
                1.0
            } else {
                ((i_i - 1 - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };
            let mesh_floor_y_prev = gutter.position.y + (total_drop / 2.0) - t_prev * total_drop;
            let floor_j_prev = (mesh_floor_y_prev / cell_size).floor() as i32;

            // Use the HIGHEST of prev, current, and next floor_j to prevent staircase gaps
            let effective_floor_j = floor_j.max(floor_j_next).max(floor_j_prev);
            let wall_top_j = effective_floor_j + wall_height_cells;

            // Is this position past the channel outlet? (in the margin zone leading to grid edge)
            let past_outlet = i >= i_end;

            for k in 0..depth {
                let k_i = k as i32;
                let in_channel_width = k >= k_start && k < k_end;
                let in_channel_length = i >= i_start && i < i_end;

                // For the margin zone past the outlet, use outlet dimensions
                let in_outlet_chute = past_outlet
                    && k_i >= (center_k - outlet_half_wid_cells)
                    && k_i < (center_k + outlet_half_wid_cells);

                // Outlet floor_j computed from mesh position at t=1.0
                let outlet_floor_j =
                    ((gutter.position.y - total_drop / 2.0) / cell_size).floor() as i32;

                for j in 0..height {
                    let j_i = j as i32;

                    // Floor - fill ALL cells at and below effective_floor_j within channel
                    // Use effective_floor_j for channel, outlet_floor_j for outlet chute
                    let is_channel_floor =
                        in_channel_length && in_channel_width && j_i <= effective_floor_j;
                    let is_outlet_chute_floor = in_outlet_chute && j_i <= outlet_floor_j;
                    let is_floor = is_channel_floor || is_outlet_chute_floor;

                    // Side walls - mark ALL cells outside channel as solid (not just thin strip)
                    // This prevents particles from escaping through gaps at grid edges
                    // Also extend walls into margin zone past outlet
                    let at_left_wall = k_i < (center_k - half_wid_cells);
                    let at_right_wall = k_i >= (center_k + half_wid_cells);
                    let is_side_wall_channel = (at_left_wall || at_right_wall)
                        && in_channel_length
                        && j_i <= wall_top_j
                        && j_i >= 0;

                    // Walls in the margin zone past outlet (using outlet width)
                    let at_left_wall_outlet = k_i < (center_k - outlet_half_wid_cells);
                    let at_right_wall_outlet = k_i >= (center_k + outlet_half_wid_cells);
                    let is_side_wall_outlet = (at_left_wall_outlet || at_right_wall_outlet)
                        && past_outlet
                        && j_i <= outlet_floor_j + wall_height_cells
                        && j_i >= 0;

                    let is_side_wall = is_side_wall_channel || is_side_wall_outlet;

                    // Back wall at inlet (left end, outside channel)
                    // Use inlet width for back wall
                    let at_back = i_i >= (center_i - half_len_cells - wall_thick_cells)
                        && i_i < (center_i - half_len_cells);
                    let is_back_wall = at_back
                        && j_i <= wall_top_j
                        && j_i >= 0
                        && k_i >= (center_k - inlet_half_wid_cells - wall_thick_cells)
                        && k_i < (center_k + inlet_half_wid_cells + wall_thick_cells);

                    if is_floor || is_side_wall || is_back_wall {
                        sim.grid.set_solid(i, j, k);
                    }
                }
            }
        }
    }

    /// Mark sluice solid cells using cell-index approach.
    pub fn mark_sluice_solid_cells(
        sim: &mut FlipSimulation3D,
        sluice: &SluicePiece,
        cell_size: f32,
    ) {
        let width = sim.grid.width;
        let height = sim.grid.height;
        let depth = sim.grid.depth;

        // Sluice position gives center in X/Z and base floor height in Y
        let center_i = (sluice.position.x / cell_size).round() as i32;
        let center_k = (sluice.position.z / cell_size).round() as i32;

        // Channel dimensions in cells
        let half_len_cells = ((sluice.length / 2.0) / cell_size).ceil() as i32;
        let half_wid_cells = ((sluice.width / 2.0) / cell_size).ceil() as i32;

        // Floor height drop due to slope
        let slope_rad = sluice.slope_deg.to_radians();
        let total_drop = sluice.length * slope_rad.tan();

        // Riffle parameters in cells
        let riffle_spacing_cells = (sluice.riffle_spacing / cell_size).round() as i32;
        let riffle_height_cells = (sluice.riffle_height / cell_size).ceil() as i32;
        let riffle_thick_cells = 2_i32;

        // Wall parameters
        let wall_height_cells = 12_i32; // Enough wall height

        // Channel bounds
        let i_start = (center_i - half_len_cells).max(0) as usize;
        let i_end = ((center_i + half_len_cells) as usize).min(width);
        let k_start = (center_k - half_wid_cells).max(0) as usize;
        let k_end = ((center_k + half_wid_cells) as usize).min(depth);

        // Inlet floor_j computed from mesh position at t=0
        let inlet_floor_j = ((sluice.position.y + total_drop / 2.0) / cell_size).floor() as i32;

        for i in 0..width {
            let i_i = i as i32;

            // Calculate t along sluice (0 = inlet, 1 = outlet)
            let t = if i_i <= center_i - half_len_cells {
                0.0
            } else if i_i >= center_i + half_len_cells {
                1.0
            } else {
                ((i_i - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };

            // Compute floor_j from mesh floor position to align with visual
            let mesh_floor_y = sluice.position.y + (total_drop / 2.0) - t * total_drop;
            let floor_j = (mesh_floor_y / cell_size).floor() as i32;

            // Look BOTH forward AND backward to cover staircase transitions
            let t_next = if i_i + 1 >= center_i + half_len_cells {
                1.0
            } else if i_i + 1 <= center_i - half_len_cells {
                0.0
            } else {
                ((i_i + 1 - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };
            let mesh_floor_y_next = sluice.position.y + (total_drop / 2.0) - t_next * total_drop;
            let floor_j_next = (mesh_floor_y_next / cell_size).floor() as i32;

            let t_prev = if i_i - 1 <= center_i - half_len_cells {
                0.0
            } else if i_i - 1 >= center_i + half_len_cells {
                1.0
            } else {
                ((i_i - 1 - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };
            let mesh_floor_y_prev = sluice.position.y + (total_drop / 2.0) - t_prev * total_drop;
            let floor_j_prev = (mesh_floor_y_prev / cell_size).floor() as i32;

            // Use HIGHEST of prev, current, and next to prevent staircase gaps
            let effective_floor_j = floor_j.max(floor_j_next).max(floor_j_prev);
            let wall_top_j = effective_floor_j + riffle_height_cells + wall_height_cells;

            // Check if this i position is on a riffle
            let dist_from_start = i_i - (center_i - half_len_cells);
            let is_riffle_x = if riffle_spacing_cells > 0 && dist_from_start > 4 {
                (dist_from_start % riffle_spacing_cells) < riffle_thick_cells
            } else {
                false
            };

            // Is this position before the channel inlet? (inlet chute zone)
            let before_inlet = (i as i32) < (center_i - half_len_cells);

            for k in 0..depth {
                let k_i = k as i32;
                let in_channel_width = k >= k_start && k < k_end;
                let in_channel_length = i >= i_start && i < i_end;

                // Inlet chute: extend floor from grid edge to channel inlet
                let in_inlet_chute = before_inlet
                    && k_i >= (center_k - half_wid_cells)
                    && k_i < (center_k + half_wid_cells);

                for j in 0..height {
                    let j_i = j as i32;

                    // Floor - fill ALL cells at and below effective_floor_j within channel
                    let is_channel_floor =
                        j_i <= effective_floor_j && in_channel_length && in_channel_width;

                    // Inlet chute floor - use inlet floor height
                    let is_inlet_chute_floor = in_inlet_chute && j_i <= inlet_floor_j;

                    let is_floor = is_channel_floor || is_inlet_chute_floor;

                    // Riffles - extend above floor at regular intervals
                    let is_riffle = is_riffle_x
                        && in_channel_width
                        && in_channel_length
                        && j_i > effective_floor_j
                        && j_i <= effective_floor_j + riffle_height_cells;

                    // Side walls - mark ALL cells outside channel as solid (not just thin strip)
                    // This prevents particles from escaping through gaps at grid edges
                    // Also extend walls into inlet chute zone
                    let at_left_wall = k_i < (center_k - half_wid_cells);
                    let at_right_wall = k_i >= (center_k + half_wid_cells);
                    let is_side_wall_channel = (at_left_wall || at_right_wall)
                        && in_channel_length
                        && j_i <= wall_top_j
                        && j_i >= 0;
                    let is_side_wall_inlet = (at_left_wall || at_right_wall)
                        && before_inlet
                        && j_i <= inlet_floor_j + wall_height_cells
                        && j_i >= 0;
                    let is_side_wall = is_side_wall_channel || is_side_wall_inlet;

                    // Back wall at inlet - NO back wall since we have inlet chute now
                    // Particles enter from the inlet side

                    if is_floor || is_riffle || is_side_wall {
                        sim.grid.set_solid(i, j, k);
                    }
                }
            }
        }
    }

    /// Mark shaker deck solid cells - walls only (grid is porous).
    /// Supports variable width (funnel effect).
    /// Marks side walls, back wall, and grate bars as solid.
    pub fn mark_shaker_deck_solid_cells(
        sim: &mut FlipSimulation3D,
        deck: &ShakerDeckPiece,
        cell_size: f32,
    ) {
        let width = sim.grid.width;
        let height = sim.grid.height;
        let depth = sim.grid.depth;

        // Deck position gives center in X/Z and base height in Y
        let center_i = (deck.position.x / cell_size).round() as i32;
        let center_k = (deck.position.z / cell_size).round() as i32;

        // Channel dimensions in cells
        let half_len_cells = ((deck.length / 2.0) / cell_size).ceil() as i32;
        // Variable width: inlet vs outlet
        let inlet_half_wid_cells = ((deck.width / 2.0) / cell_size).ceil() as i32;
        let outlet_half_wid_cells = ((deck.end_width / 2.0) / cell_size).ceil() as i32;

        // Floor height drop due to tilt
        let tilt_rad = deck.tilt_deg.to_radians();
        let total_drop = deck.length * tilt_rad.tan();

        // Wall parameters - deck walls are solid
        let wall_height_cells = ((deck.wall_height / cell_size).ceil() as i32).max(4);
        let wall_thick_cells = 2_i32;

        // Grate parameters - bars and holes
        let bar_spacing = deck.hole_size + deck.bar_thickness;
        let bar_thick_cells = (deck.bar_thickness / cell_size).ceil() as i32;
        let hole_cells = (deck.hole_size / cell_size).floor() as i32;
        let pattern_cells = bar_thick_cells + hole_cells;

        // Cross bar spacing (every 3rd bar_spacing)
        let cross_spacing = bar_spacing * 3.0;
        let cross_pattern_cells = (cross_spacing / cell_size).round() as i32;

        // Channel bounds in i (length direction)
        let i_start = (center_i - half_len_cells).max(0) as usize;
        let i_end = ((center_i + half_len_cells) as usize).min(width);

        for i in 0..width {
            let i_i = i as i32;

            // Calculate position along deck (0.0 = inlet, 1.0 = outlet)
            let t = if i_i <= center_i - half_len_cells {
                0.0
            } else if i_i >= center_i + half_len_cells {
                1.0
            } else {
                ((i_i - (center_i - half_len_cells)) as f32)
                    / ((half_len_cells * 2) as f32).max(1.0)
            };

            // Variable width at this position (funnel effect)
            let local_half_wid_cells = (inlet_half_wid_cells as f32
                + (outlet_half_wid_cells - inlet_half_wid_cells) as f32 * t)
                as i32;

            // Compute floor_j from mesh floor position
            let mesh_floor_y = deck.position.y + (total_drop / 2.0) - t * total_drop;
            let floor_j = (mesh_floor_y / cell_size).floor() as i32;
            let wall_top_j = floor_j + wall_height_cells;

            let in_channel_length = i >= i_start && i < i_end;

            // Check if this X position is on a cross bar
            let dist_from_start = i_i - (center_i - half_len_cells);
            let on_cross_bar = if cross_pattern_cells > 0 {
                (dist_from_start % cross_pattern_cells) < bar_thick_cells
            } else {
                false
            };

            for k in 0..depth {
                let k_i = k as i32;

                // Check if this Z position is on a longitudinal bar
                let dist_from_center_z = (k_i - center_k).abs();
                let on_long_bar = if pattern_cells > 0 {
                    (dist_from_center_z % pattern_cells) < bar_thick_cells
                } else {
                    true // If no pattern, treat as solid
                };

                // Within channel width?
                let in_channel_width = k_i >= (center_k - local_half_wid_cells)
                    && k_i < (center_k + local_half_wid_cells);

                for j in 0..height {
                    let j_i = j as i32;

                    // Side walls - mark cells outside channel width as solid
                    let at_left_wall = k_i < (center_k - local_half_wid_cells);
                    let at_right_wall = k_i >= (center_k + local_half_wid_cells);
                    let is_side_wall = (at_left_wall || at_right_wall)
                        && in_channel_length
                        && j_i <= wall_top_j
                        && j_i >= floor_j;

                    // Back wall at inlet
                    let at_back = i_i >= (center_i - half_len_cells - wall_thick_cells)
                        && i_i < (center_i - half_len_cells);
                    let is_back_wall = at_back
                        && j_i <= wall_top_j
                        && j_i >= floor_j
                        && k_i >= (center_k - inlet_half_wid_cells - wall_thick_cells)
                        && k_i < (center_k + inlet_half_wid_cells + wall_thick_cells);

                    // Grate bars - at floor level, where bars exist (not holes)
                    // Bars are solid, holes let particles through
                    // Make bars 4 cells thick to prevent tunneling
                    let at_floor_level = j_i >= floor_j && j_i <= floor_j + 3;
                    let is_grate_bar = in_channel_length
                        && in_channel_width
                        && at_floor_level
                        && (on_long_bar || on_cross_bar);

                    if is_side_wall || is_back_wall || is_grate_bar {
                        sim.grid.set_solid(i, j, k);
                    }
                }
            }
        }
    }
}
