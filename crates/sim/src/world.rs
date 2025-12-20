//! World - manages chunks and cross-chunk operations.

use crate::chunk::{Chunk, CHUNK_SIZE};
use crate::fluid;
use crate::material::Material;
use crate::water;
use rustc_hash::{FxHashMap, FxHashSet};

/// The game world - an infinite grid of chunks.
pub struct World {
    pub chunks: FxHashMap<(i32, i32), Chunk>,
    pub seed: u64,
    pub frame: u64,
    /// Cells that need to be processed this frame
    active_cells: FxHashSet<(i32, i32)>,
    /// Cells to process next frame (built during current frame)
    next_active: FxHashSet<(i32, i32)>,
    /// Water cells that have been static (for auto-leveling)
    static_water_frames: FxHashMap<(i32, i32), u8>,
}

impl World {
    pub fn new(seed: u64) -> Self {
        Self {
            chunks: FxHashMap::default(),
            seed,
            frame: 0,
            active_cells: FxHashSet::default(),
            next_active: FxHashSet::default(),
            static_water_frames: FxHashMap::default(),
        }
    }

    /// Mark a cell and its neighbors as active for next frame.
    fn mark_active(&mut self, wx: i32, wy: i32) {
        // Mark cell and all 8 neighbors
        for dy in -1..=1 {
            for dx in -1..=1 {
                self.next_active.insert((wx + dx, wy + dy));
            }
        }
    }

    /// Initialize active cells for a region (call after setting up world).
    pub fn activate_region(&mut self, min_x: i32, min_y: i32, max_x: i32, max_y: i32) {
        for y in min_y..max_y {
            for x in min_x..max_x {
                let mat = self.get_material(x, y);
                // Only activate non-static materials
                if mat != Material::Air && mat != Material::Rock {
                    self.active_cells.insert((x, y));
                    // Also activate neighbors (they might flow into this space)
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            self.active_cells.insert((x + dx, y + dy));
                        }
                    }
                }
            }
        }
    }

    /// Get or create chunk at chunk coordinates.
    pub fn get_or_create_chunk(&mut self, cx: i32, cy: i32) -> &mut Chunk {
        self.chunks.entry((cx, cy)).or_insert_with(Chunk::new)
    }

    /// Get chunk if it exists.
    pub fn get_chunk(&self, cx: i32, cy: i32) -> Option<&Chunk> {
        self.chunks.get(&(cx, cy))
    }

    /// Get mutable chunk if it exists.
    pub fn get_chunk_mut(&mut self, cx: i32, cy: i32) -> Option<&mut Chunk> {
        self.chunks.get_mut(&(cx, cy))
    }

    /// Convert world coordinates to chunk coordinates.
    #[inline]
    pub fn world_to_chunk(wx: i32, wy: i32) -> (i32, i32) {
        (
            wx.div_euclid(CHUNK_SIZE as i32),
            wy.div_euclid(CHUNK_SIZE as i32),
        )
    }

    /// Convert world coordinates to local chunk coordinates.
    #[inline]
    pub fn world_to_local(wx: i32, wy: i32) -> (usize, usize) {
        (
            wx.rem_euclid(CHUNK_SIZE as i32) as usize,
            wy.rem_euclid(CHUNK_SIZE as i32) as usize,
        )
    }

    /// Get material at world coordinates.
    pub fn get_material(&self, wx: i32, wy: i32) -> Material {
        let (cx, cy) = Self::world_to_chunk(wx, wy);
        let (lx, ly) = Self::world_to_local(wx, wy);

        self.chunks
            .get(&(cx, cy))
            .map(|chunk| chunk.get_material(lx, ly))
            .unwrap_or(Material::Air)
    }

    /// Set material at world coordinates.
    pub fn set_material(&mut self, wx: i32, wy: i32, material: Material) {
        let (cx, cy) = Self::world_to_chunk(wx, wy);
        let (lx, ly) = Self::world_to_local(wx, wy);

        let chunk = self.get_or_create_chunk(cx, cy);
        chunk.set_material(lx, ly, material);

        // Mark cell and neighbors as active IMMEDIATELY (for this frame)
        if material != Material::Rock {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    self.active_cells.insert((wx + dx, wy + dy));
                }
            }
        }
    }

    /// Get water mass at world coordinates.
    pub fn get_water(&self, wx: i32, wy: i32) -> f32 {
        let (cx, cy) = Self::world_to_chunk(wx, wy);
        let (lx, ly) = Self::world_to_local(wx, wy);

        self.chunks
            .get(&(cx, cy))
            .map(|chunk| chunk.get_water(lx, ly))
            .unwrap_or(0.0)
    }

    /// Set water mass at world coordinates.
    pub fn set_water(&mut self, wx: i32, wy: i32, amount: f32) {
        let (cx, cy) = Self::world_to_chunk(wx, wy);
        let (lx, ly) = Self::world_to_local(wx, wy);

        let chunk = self.get_or_create_chunk(cx, cy);
        chunk.set_water(lx, ly, amount);
    }

    /// Add water mass at world coordinates.
    pub fn add_water(&mut self, wx: i32, wy: i32, amount: f32) {
        let (cx, cy) = Self::world_to_chunk(wx, wy);
        let (lx, ly) = Self::world_to_local(wx, wy);

        let chunk = self.get_or_create_chunk(cx, cy);
        // Only add water if the cell isn't solid
        if !chunk.get_material(lx, ly).is_solid() {
            chunk.add_water(lx, ly, amount);
        }
    }

    /// Get water velocity at world coordinates (for particle suspension).
    pub fn get_water_velocity(&self, wx: i32, wy: i32) -> (f32, f32) {
        let (cx, cy) = Self::world_to_chunk(wx, wy);
        let (lx, ly) = Self::world_to_local(wx, wy);

        self.chunks
            .get(&(cx, cy))
            .map(|chunk| water::get_water_velocity(chunk, lx, ly))
            .unwrap_or((0.0, 0.0))
    }

    /// Set particle velocity at world coordinates (makes particles flow with water).
    pub fn set_particle_velocity(&mut self, wx: i32, wy: i32, vx: f32, vy: f32) {
        let (cx, cy) = Self::world_to_chunk(wx, wy);
        let (lx, ly) = Self::world_to_local(wx, wy);

        if let Some(chunk) = self.chunks.get_mut(&(cx, cy)) {
            let idx = Chunk::index(lx, ly);
            chunk.vel_x[idx] = vx;
            chunk.vel_y[idx] = vy;
        }
    }

    /// Check if a cell has been updated this frame (world coordinates).
    pub fn is_updated(&self, wx: i32, wy: i32) -> bool {
        let (cx, cy) = Self::world_to_chunk(wx, wy);
        let (lx, ly) = Self::world_to_local(wx, wy);

        self.chunks
            .get(&(cx, cy))
            .map(|chunk| chunk.is_updated(lx, ly))
            .unwrap_or(false)
    }

    /// Mark a cell as updated this frame (world coordinates).
    pub fn mark_updated(&mut self, wx: i32, wy: i32) {
        let (cx, cy) = Self::world_to_chunk(wx, wy);
        let (lx, ly) = Self::world_to_local(wx, wy);

        if let Some(chunk) = self.chunks.get_mut(&(cx, cy)) {
            chunk.mark_updated(lx, ly);
        }
    }

    /// Set material AND velocity at world coordinates.
    pub fn set_state(&mut self, wx: i32, wy: i32, material: Material, vx: f32, vy: f32) {
        let (cx, cy) = Self::world_to_chunk(wx, wy);
        let (lx, ly) = Self::world_to_local(wx, wy);

        let chunk = self.get_or_create_chunk(cx, cy);
        chunk.set_state(lx, ly, material, vx, vy);
        
        // Mark active
        self.mark_active(wx, wy);
    }

    /// Swap two cells at world coordinates.
    pub fn swap(&mut self, wx1: i32, wy1: i32, wx2: i32, wy2: i32) {
        let mat1 = self.get_material(wx1, wy1);
        let mat2 = self.get_material(wx2, wy2);

        // Direct set without triggering mark_active (we'll do it manually)
        let (cx1, cy1) = Self::world_to_chunk(wx1, wy1);
        let (lx1, ly1) = Self::world_to_local(wx1, wy1);
        let (cx2, cy2) = Self::world_to_chunk(wx2, wy2);
        let (lx2, ly2) = Self::world_to_local(wx2, wy2);

        if let Some(chunk) = self.chunks.get_mut(&(cx1, cy1)) {
            chunk.set_material(lx1, ly1, mat2);
        }
        if let Some(chunk) = self.chunks.get_mut(&(cx2, cy2)) {
            chunk.set_material(lx2, ly2, mat1);
        }

        self.mark_updated(wx1, wy1);
        self.mark_updated(wx2, wy2);

        // Mark both cells and neighbors as active for next frame
        self.mark_active(wx1, wy1);
        self.mark_active(wx2, wy2);
    }

    /// Update simulation for a region (world coordinates).
    /// Only processes active cells for performance.
    pub fn update_region(&mut self, min_wx: i32, min_wy: i32, max_wx: i32, max_wy: i32) {
        self.frame += 1;

        // Clear next_active for building this frame's results
        self.next_active.clear();

        // Find which chunks overlap the simulation region
        let min_cx = Self::world_to_chunk(min_wx, 0).0;
        let max_cx = Self::world_to_chunk(max_wx, 0).0;
        let min_cy = Self::world_to_chunk(0, min_wy).1;
        let max_cy = Self::world_to_chunk(0, max_wy).1;

        // Clear updated flags for all chunks in region
        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                if let Some(chunk) = self.chunks.get_mut(&(cx, cy)) {
                    chunk.clear_updated();
                }
            }
        }

        // UNIFIED PHYSICS: Fluid solver runs on ALL chunks (particles + water create turbulence)
        const DT: f32 = 1.0 / 60.0;
        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                if let Some(chunk) = self.chunks.get_mut(&(cx, cy)) {
                    // Run Navier-Stokes fluid solver - creates velocity field for ALL materials
                    if chunk.is_active || chunk.has_water {
                        crate::fluid::solve_fluid(chunk);
                    }
                }
            }
        }

        // Update water and particle transport (mass-based virtual pipes + velocity field)
        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                if let Some(chunk) = self.chunks.get_mut(&(cx, cy)) {
                    if chunk.has_water || chunk.is_active {
                        water::update_water(chunk, DT);
                    }
                }
            }
        }

        // Handle cross-chunk water flow
        self.update_cross_chunk_water(min_cx, min_cy, max_cx, max_cy, DT);

        // Collect and sort active cells (bottom-to-top, with x alternation)
        let mut cells: Vec<(i32, i32)> = self.active_cells
            .iter()
            .filter(|(x, y)| *x >= min_wx && *x < max_wx && *y >= min_wy && *y < max_wy)
            .copied()
            .collect();

        // Sort by y descending (bottom first), then x based on frame
        let frame = self.frame;
        cells.sort_by(|a, b| {
            let y_cmp = b.1.cmp(&a.1); // y descending (bottom first)
            if y_cmp != std::cmp::Ordering::Equal {
                y_cmp
            } else if frame & 1 == 0 {
                a.0.cmp(&b.0) // x ascending on even frames
            } else {
                b.0.cmp(&a.0) // x descending on odd frames
            }
        });

        // Process only active cells
        for (wx, wy) in cells {
            if self.is_updated(wx, wy) {
                continue;
            }

            let material = self.get_material(wx, wy);
            let water_level = self.get_water(wx, wy);

            // UNIFIED PHYSICS: If particle is submerged, water system handles it
            // Only run old CA physics for DRY particles
            let moved = match material {
                Material::Air | Material::Rock => false,
                _ if water_level > 0.1 => false, // In water - water.rs handles movement
                _ if material.is_powder() => self.update_powder(wx, wy),
                _ if material.is_liquid() => self.update_liquid(wx, wy),
                _ => false,
            };

            if !moved {
                self.mark_updated(wx, wy);
            }
        }

        // Swap active sets for next frame
        std::mem::swap(&mut self.active_cells, &mut self.next_active);
    }

    /// Auto-level water bodies that have been static for several frames.
    fn auto_level_water(&mut self, min_wx: i32, min_wy: i32, max_wx: i32, max_wy: i32) {
        const STATIC_THRESHOLD: u8 = 4; // Checks before auto-leveling (~8 frames)

        // Find water surface cells and track static frames
        let mut surfaces_to_check: Vec<(i32, i32)> = Vec::new();
        let mut to_remove: Vec<(i32, i32)> = Vec::new();

        // Scan for water surfaces in region
        for y in min_wy..max_wy {
            for x in min_wx..max_wx {
                if self.get_material(x, y) == Material::Water {
                    let above = self.get_material(x, y - 1);
                    if above != Material::Water {
                        // This is a surface cell
                        if !self.active_cells.contains(&(x, y)) {
                            // Not active = static, increment counter
                            let count = self.static_water_frames.entry((x, y)).or_insert(0);
                            *count = count.saturating_add(1);
                            if *count >= STATIC_THRESHOLD {
                                surfaces_to_check.push((x, y));
                            }
                        } else {
                            // Active = moving, reset counter
                            to_remove.push((x, y));
                        }
                    }
                }
            }
        }

        // Remove counters for active water
        for pos in to_remove {
            self.static_water_frames.remove(&pos);
        }

        // Process static water bodies
        let mut leveled: FxHashSet<(i32, i32)> = FxHashSet::default();
        for (sx, sy) in surfaces_to_check {
            if leveled.contains(&(sx, sy)) {
                continue;
            }

            // Find connected water body via flood fill on surface
            let body = self.find_water_body_surface(sx, sy, min_wx, max_wx);
            if body.len() < 2 {
                continue;
            }

            // Check if surface is already level (all same y)
            let min_y = body.iter().map(|(_, y)| *y).min().unwrap();
            let max_y = body.iter().map(|(_, y)| *y).max().unwrap();

            if min_y == max_y {
                // Already level, clear static counters
                for pos in &body {
                    self.static_water_frames.remove(pos);
                }
                continue;
            }

            // Not level - flatten it!
            self.flatten_water_body(&body);

            // Mark as leveled and clear counters
            for pos in &body {
                leveled.insert(*pos);
                self.static_water_frames.remove(pos);
            }
        }
    }

    /// Find connected water body and return surface cells.
    /// Uses proper flood fill through water to handle varying heights.
    fn find_water_body_surface(&self, start_x: i32, start_y: i32, min_wx: i32, max_wx: i32) -> Vec<(i32, i32)> {
        const MAX_CELLS: usize = 5000; // Limit to prevent slow flood fills

        let mut visited: FxHashSet<(i32, i32)> = FxHashSet::default();
        let mut surfaces: FxHashMap<i32, i32> = FxHashMap::default(); // x -> highest y (surface)
        let mut queue = vec![(start_x, start_y)];

        // Flood fill through all connected water
        while let Some((x, y)) = queue.pop() {
            if visited.len() >= MAX_CELLS {
                break;
            }
            if x < min_wx || x >= max_wx {
                continue;
            }
            if visited.contains(&(x, y)) {
                continue;
            }
            if self.get_material(x, y) != Material::Water {
                continue;
            }

            visited.insert((x, y));

            // Track the highest (smallest y) water cell at each x = surface
            surfaces.entry(x)
                .and_modify(|sy| *sy = (*sy).min(y))
                .or_insert(y);

            // Add all 4 neighbors (not diagonal - water connects orthogonally)
            queue.push((x - 1, y));
            queue.push((x + 1, y));
            queue.push((x, y - 1));
            queue.push((x, y + 1));
        }

        // Convert to surface list
        surfaces.into_iter().map(|(x, y)| (x, y)).collect()
    }

    /// Gradually level a water body by equalizing surface Y coordinates.
    /// Only moves water DOWNWARD (from high surface to low surface).
    fn flatten_water_body(&mut self, surface_cells: &[(i32, i32)]) {
        if surface_cells.len() < 2 {
            return;
        }

        // Sort by surface Y - smallest Y = highest on screen
        let mut surfaces: Vec<(i32, i32)> = surface_cells.to_vec();
        surfaces.sort_by_key(|(_, y)| *y);

        let (_, high_y) = surfaces[0]; // Highest surface (smallest Y)
        let (_, low_y) = surfaces[surfaces.len() - 1]; // Lowest surface (largest Y)

        // Only level if surfaces differ by more than 1
        if low_y <= high_y + 1 {
            return;
        }

        // Move up to 5 cells per call
        for _ in 0..5 {
            // Re-check surfaces after each move
            let high_surface = surfaces[0].1;
            let low_surface = surfaces[surfaces.len() - 1].1;

            if low_surface <= high_surface + 1 {
                break;
            }

            let (from_x, from_y) = surfaces[0];
            let (to_x, to_y) = surfaces[surfaces.len() - 1];

            // Only proceed if target is actually BELOW source (larger Y)
            if to_y <= from_y {
                break;
            }

            // Remove water from high surface
            self.set_material_direct(from_x, from_y, Material::Air);

            // Add water above low surface
            let target_y = to_y - 1;
            if self.get_material(to_x, target_y) == Material::Air {
                self.set_material_direct(to_x, target_y, Material::Water);

                // Update surfaces for next iteration
                surfaces[0] = (from_x, from_y + 1);
                let last = surfaces.len() - 1;
                surfaces[last] = (to_x, target_y);

                // Re-sort
                surfaces.sort_by_key(|(_, y)| *y);

                // Mark as active
                for (x, y) in [(from_x, from_y), (to_x, target_y)] {
                    let (cx, cy) = Self::world_to_chunk(x, y);
                    if let Some(chunk) = self.chunks.get_mut(&(cx, cy)) {
                        chunk.needs_render = true;
                    }
                    self.active_cells.insert((x, y));
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            self.active_cells.insert((x + dx, y + dy));
                        }
                    }
                }
            } else {
                // Can't place - restore and stop
                self.set_material_direct(from_x, from_y, Material::Water);
                break;
            }
        }
    }

    /// Set material without triggering active cell tracking (for bulk operations).
    fn set_material_direct(&mut self, wx: i32, wy: i32, material: Material) {
        let (cx, cy) = Self::world_to_chunk(wx, wy);
        let (lx, ly) = Self::world_to_local(wx, wy);

        if let Some(chunk) = self.chunks.get_mut(&(cx, cy)) {
            chunk.set_material(lx, ly, material);
        }
    }

    /// Update all loaded chunks (convenience method, slower).
    pub fn update(&mut self) {
        let chunk_keys: Vec<_> = self.chunks.keys().copied().collect();
        if chunk_keys.is_empty() {
            return;
        }

        let min_cx = chunk_keys.iter().map(|k| k.0).min().unwrap();
        let max_cx = chunk_keys.iter().map(|k| k.0).max().unwrap();
        let min_cy = chunk_keys.iter().map(|k| k.1).min().unwrap();
        let max_cy = chunk_keys.iter().map(|k| k.1).max().unwrap();

        self.update_region(
            min_cx * CHUNK_SIZE as i32,
            min_cy * CHUNK_SIZE as i32,
            (max_cx + 1) * CHUNK_SIZE as i32,
            (max_cy + 1) * CHUNK_SIZE as i32,
        );
    }

    /// Update powder material (Soil, Gold).
    /// Uses water physics: particles are dense masses that flow with water.
    fn update_powder(&mut self, wx: i32, wy: i32) -> bool {
        let material = self.get_material(wx, wy);
        let density = material.density() as f32;
        let water_here = self.get_water(wx, wy);

        // Get water flow velocity at this cell
        let (flow_x, flow_y) = self.get_water_velocity(wx, wy);
        let flow_speed = (flow_x * flow_x + flow_y * flow_y).sqrt();

        // Threshold: denser particles need stronger flow to move horizontally
        // Water density is 10, so scale threshold by particle density
        let flow_threshold = density * 0.05;

        // 1. If submerged and flow is strong enough, move with water
        if water_here > 0.3 && flow_speed > flow_threshold {
            // Move in dominant flow direction
            let dx = if flow_x.abs() > 0.5 { flow_x.signum() as i32 } else { 0 };
            let dy = if flow_y > 0.5 { 1 } else { 0 }; // Only flow down, not up

            if dx != 0 || dy != 0 {
                let nx = wx + dx;
                let ny = wy + dy;
                let neighbor = self.get_material(nx, ny);

                // Can flow into air or lighter materials
                if neighbor == Material::Air || (!neighbor.is_solid() && density > neighbor.density() as f32) {
                    self.swap(wx, wy, nx, ny);
                    return true;
                }
            }
        }

        // 2. Gravity: fall straight down - sink through air or lighter materials
        let below = self.get_material(wx, wy + 1);
        if below == Material::Air || (!below.is_solid() && density > below.density() as f32) {
            self.swap(wx, wy, wx, wy + 1);
            return true;
        }

        // 3. Slide diagonally down if blocked
        let try_left_first = ((wx ^ wy) as u64 ^ self.frame) & 1 == 0;
        let (dx1, dx2) = if try_left_first { (-1, 1) } else { (1, -1) };

        for dx in [dx1, dx2] {
            let nx = wx + dx;
            let diag = self.get_material(nx, wy + 1);
            if diag == Material::Air || (!diag.is_solid() && density > diag.density() as f32) {
                self.swap(wx, wy, nx, wy + 1);
                return true;
            }
        }

        false
    }

    /// Update liquid material (Water, Mud).
    /// Uses water physics: liquids are masses that flow via pressure differential.
    /// Mud is denser than water so it sinks, but still flows with water currents.
    fn update_liquid(&mut self, wx: i32, wy: i32) -> bool {
        let material = self.get_material(wx, wy);
        let density = material.density() as f32;
        let water_here = self.get_water(wx, wy);

        // Get water flow velocity
        let (flow_x, flow_y) = self.get_water_velocity(wx, wy);

        // 1. Flow with water current (Mud gets carried by water)
        if water_here > 0.2 {
            // Mud is denser so needs stronger flow, but not as much as powders
            let flow_threshold = density * 0.02;
            let flow_speed = (flow_x * flow_x + flow_y * flow_y).sqrt();

            if flow_speed > flow_threshold {
                let dx = if flow_x.abs() > 0.3 { flow_x.signum() as i32 } else { 0 };
                let dy = if flow_y > 0.3 { 1 } else { 0 };

                if dx != 0 || dy != 0 {
                    let nx = wx + dx;
                    let ny = wy + dy;
                    let neighbor = self.get_material(nx, ny);

                    if neighbor == Material::Air || (!neighbor.is_solid() && density > neighbor.density() as f32) {
                        self.swap(wx, wy, nx, ny);
                        return true;
                    }
                }
            }
        }

        // 2. Gravity: fall straight down
        let below = self.get_material(wx, wy + 1);
        if below == Material::Air || (!below.is_solid() && density > below.density() as f32) {
            self.swap(wx, wy, wx, wy + 1);
            return true;
        }

        // 3. Slide diagonally if falling is blocked
        let try_left_first = ((wx ^ wy) as u64 ^ self.frame) & 1 == 0;
        let (dx1, dx2) = if try_left_first { (-1, 1) } else { (1, -1) };

        for dx in [dx1, dx2] {
            let diag = self.get_material(wx + dx, wy + 1);
            if diag == Material::Air || (!diag.is_solid() && density > diag.density() as f32) {
                self.swap(wx, wy, wx + dx, wy + 1);
                return true;
            }
        }

        // 4. Horizontal spread (viscous flow) - mud spreads slowly
        let spread_rate = material.spread_rate() as i32;
        for dx in [dx1, dx2] {
            for dist in 1..=spread_rate {
                let nx = wx + dx * dist;
                let neighbor = self.get_material(nx, wy);

                if neighbor.is_solid() {
                    break;
                }

                if neighbor == Material::Air {
                    self.swap(wx, wy, nx, wy);
                    return true;
                }
            }
        }

        false
    }

    /// Swap two cells, preserving velocity field (for fluid simulation).
    fn swap_with_velocity(&mut self, wx1: i32, wy1: i32, wx2: i32, wy2: i32) {
        let mat1 = self.get_material(wx1, wy1);
        let mat2 = self.get_material(wx2, wy2);

        let (cx1, cy1) = Self::world_to_chunk(wx1, wy1);
        let (lx1, ly1) = Self::world_to_local(wx1, wy1);
        let (cx2, cy2) = Self::world_to_chunk(wx2, wy2);
        let (lx2, ly2) = Self::world_to_local(wx2, wy2);

        // Get velocities before swap
        let (vx1, vy1) = if let Some(chunk) = self.chunks.get(&(cx1, cy1)) {
            let idx = Chunk::index(lx1, ly1);
            (chunk.vel_x[idx], chunk.vel_y[idx])
        } else {
            (0.0, 0.0)
        };

        // Swap materials only (velocity stays on grid for Eulerian simulation)
        if let Some(chunk) = self.chunks.get_mut(&(cx1, cy1)) {
            chunk.materials[Chunk::index(lx1, ly1)] = mat2;
            chunk.needs_render = true;
        }
        if let Some(chunk) = self.chunks.get_mut(&(cx2, cy2)) {
            chunk.materials[Chunk::index(lx2, ly2)] = mat1;
            chunk.needs_render = true;
        }

        self.mark_updated(wx1, wy1);
        self.mark_updated(wx2, wy2);
        self.mark_active(wx1, wy1);
        self.mark_active(wx2, wy2);
    }

    /// Get column height at world coordinates.
    fn get_column_height(&self, wx: i32, wy: i32) -> i32 {
        let mut height = 0;
        let mut y = wy;
        while self.get_material(wx, y) == Material::Water {
            height += 1;
            y += 1;
            if height > 500 {
                break;
            }
        }
        height
    }

    /// Find surface (topmost water) at world coordinates.
    fn find_surface(&self, wx: i32, wy: i32) -> Option<i32> {
        if self.get_material(wx, wy) != Material::Water {
            return None;
        }
        let mut y = wy;
        while self.get_material(wx, y - 1) == Material::Water {
            y -= 1;
            if (wy - y) > 500 {
                break;
            }
        }
        Some(y)
    }

    /// Generate terrain for a chunk at given chunk coordinates.
    pub fn generate_chunk(&mut self, cx: i32, cy: i32) {
        // Copy seed to avoid borrow issues
        let seed = self.seed;
        let chunk = self.get_or_create_chunk(cx, cy);

        // Simple terrain: surface at y=0, everything below is soil/rock
        for ly in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                // World Y coordinate
                let wy = cy * CHUNK_SIZE as i32 + ly as i32;

                let material = if wy < 0 {
                    // Above ground = air
                    Material::Air
                } else if wy < 20 {
                    // Top layer = soil
                    Material::Soil
                } else if wy < 100 {
                    // Mix of soil and rock
                    let hash = simple_hash(cx, cy, lx as i32, ly as i32, seed);
                    if hash % 3 == 0 {
                        Material::Rock
                    } else {
                        Material::Soil
                    }
                } else {
                    // Deep = mostly rock
                    let hash = simple_hash(cx, cy, lx as i32, ly as i32, seed);
                    if hash % 5 == 0 {
                        Material::Soil
                    } else {
                        Material::Rock
                    }
                };

                chunk.set_material(lx, ly, material);
            }
        }

        // Sprinkle some gold (rare)
        for ly in 0..CHUNK_SIZE {
            for lx in 0..CHUNK_SIZE {
                let wy = cy * CHUNK_SIZE as i32 + ly as i32;
                if wy > 30 {
                    let hash = simple_hash(cx, cy, lx as i32, ly as i32, seed.wrapping_add(1));
                    // ~0.5% chance of gold in deep areas
                    if hash % 200 == 0 {
                        chunk.set_material(lx, ly, Material::Gold);
                    }
                }
            }
        }

        chunk.recalculate_active();
    }

    /// Ensure chunks exist around a world position.
    pub fn ensure_chunks_around(&mut self, wx: i32, wy: i32, radius: i32) {
        let (center_cx, center_cy) = Self::world_to_chunk(wx, wy);

        for cy in (center_cy - radius)..=(center_cy + radius) {
            for cx in (center_cx - radius)..=(center_cx + radius) {
                if !self.chunks.contains_key(&(cx, cy)) {
                    self.generate_chunk(cx, cy);
                }
            }
        }
    }

    /// Handle water flow across chunk boundaries.
    /// This transfers water mass between adjacent chunks at their edges.
    fn update_cross_chunk_water(&mut self, min_cx: i32, min_cy: i32, max_cx: i32, max_cy: i32, dt: f32) {
        const GRAVITY: f32 = 30.0;  // Match water.rs
        const CROSS_FLOW_RATE: f32 = 1.5;

        // Collect boundary flows to apply (to avoid borrow issues)
        let mut transfers: Vec<((i32, i32, usize, usize), (i32, i32, usize, usize), f32)> = Vec::new();

        // For each chunk, check right and down boundaries
        for cy in min_cy..=max_cy {
            for cx in min_cx..=max_cx {
                // Right boundary: chunk (cx, cy) edge x=CHUNK_SIZE-1 -> chunk (cx+1, cy) edge x=0
                if cx < max_cx {
                    if let (Some(chunk), Some(chunk_right)) = (
                        self.chunks.get(&(cx, cy)),
                        self.chunks.get(&(cx + 1, cy)),
                    ) {
                        for y in 0..CHUNK_SIZE {
                            let idx_left = Chunk::index(CHUNK_SIZE - 1, y);
                            let idx_right = Chunk::index(0, y);

                            let water_left = chunk.water_mass[idx_left];
                            let water_right = chunk_right.water_mass[idx_right];
                            let mat_left = chunk.materials[idx_left];
                            let mat_right = chunk_right.materials[idx_right];

                            // Skip if either side is solid
                            if mat_left.is_solid() || mat_right.is_solid() {
                                continue;
                            }

                            // Calculate flow based on pressure difference
                            let height_diff = water_left - water_right;
                            if height_diff.abs() > 0.01 {
                                let flow = height_diff * GRAVITY * dt * CROSS_FLOW_RATE;
                                // Clamp to prevent instability but allow good flow
                                let flow = flow.clamp(-water_right * 0.4, water_left * 0.4);

                                if flow.abs() > 0.001 {
                                    transfers.push((
                                        (cx, cy, CHUNK_SIZE - 1, y),
                                        (cx + 1, cy, 0, y),
                                        flow,
                                    ));
                                }
                            }
                        }
                    }
                }

                // Down boundary: chunk (cx, cy) edge y=CHUNK_SIZE-1 -> chunk (cx, cy+1) edge y=0
                if cy < max_cy {
                    if let (Some(chunk), Some(chunk_down)) = (
                        self.chunks.get(&(cx, cy)),
                        self.chunks.get(&(cx, cy + 1)),
                    ) {
                        for x in 0..CHUNK_SIZE {
                            let idx_top = Chunk::index(x, CHUNK_SIZE - 1);
                            let idx_bottom = Chunk::index(x, 0);

                            let water_top = chunk.water_mass[idx_top];
                            let water_bottom = chunk_down.water_mass[idx_bottom];
                            let mat_top = chunk.materials[idx_top];
                            let mat_bottom = chunk_down.materials[idx_bottom];

                            // Skip if either side is solid
                            if mat_top.is_solid() || mat_bottom.is_solid() {
                                continue;
                            }

                            // Calculate flow - gravity adds to downward pressure
                            let height_diff = water_top - water_bottom + 1.0; // +1.0 for stronger gravity
                            let flow = height_diff * GRAVITY * dt * CROSS_FLOW_RATE;
                            // Clamp to prevent instability but allow good flow
                            let flow = flow.clamp(-water_bottom * 0.4, water_top * 0.6);

                            if flow.abs() > 0.001 {
                                transfers.push((
                                    (cx, cy, x, CHUNK_SIZE - 1),
                                    (cx, cy + 1, x, 0),
                                    flow,
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Apply all transfers
        for ((cx1, cy1, x1, y1), (cx2, cy2, x2, y2), flow) in transfers {
            // Remove from source
            if let Some(chunk) = self.chunks.get_mut(&(cx1, cy1)) {
                let idx = Chunk::index(x1, y1);
                chunk.water_mass[idx] = (chunk.water_mass[idx] - flow).max(0.0);
                chunk.needs_render = true;
                if chunk.water_mass[idx] > 0.01 {
                    chunk.has_water = true;
                }
            }

            // Add to destination
            if let Some(chunk) = self.chunks.get_mut(&(cx2, cy2)) {
                let idx = Chunk::index(x2, y2);
                chunk.water_mass[idx] = (chunk.water_mass[idx] + flow).max(0.0);
                chunk.needs_render = true;
                if chunk.water_mass[idx] > 0.01 {
                    chunk.has_water = true;
                    chunk.is_active = true;
                }
            }
        }
    }
}

/// Simple hash function for deterministic world generation.
fn simple_hash(cx: i32, cy: i32, lx: i32, ly: i32, seed: u64) -> u64 {
    let mut h = seed;
    h = h.wrapping_mul(0x517cc1b727220a95);
    h ^= cx as u64;
    h = h.wrapping_mul(0x517cc1b727220a95);
    h ^= cy as u64;
    h = h.wrapping_mul(0x517cc1b727220a95);
    h ^= lx as u64;
    h = h.wrapping_mul(0x517cc1b727220a95);
    h ^= ly as u64;
    h = h.wrapping_mul(0x517cc1b727220a95);
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinate_conversion() {
        // Positive coordinates
        assert_eq!(World::world_to_chunk(0, 0), (0, 0));
        assert_eq!(World::world_to_chunk(63, 63), (0, 0));
        assert_eq!(World::world_to_chunk(64, 64), (1, 1));

        // Negative coordinates (important for infinite world)
        assert_eq!(World::world_to_chunk(-1, -1), (-1, -1));
        assert_eq!(World::world_to_chunk(-64, -64), (-1, -1));
        assert_eq!(World::world_to_chunk(-65, -65), (-2, -2));

        // Local coordinates
        assert_eq!(World::world_to_local(0, 0), (0, 0));
        assert_eq!(World::world_to_local(63, 63), (63, 63));
        assert_eq!(World::world_to_local(64, 64), (0, 0));
        assert_eq!(World::world_to_local(-1, -1), (63, 63));
    }

    #[test]
    fn set_and_get_material() {
        let mut world = World::new(12345);

        world.set_material(100, 100, Material::Gold);
        assert_eq!(world.get_material(100, 100), Material::Gold);

        // Negative coordinates
        world.set_material(-50, -50, Material::Water);
        assert_eq!(world.get_material(-50, -50), Material::Water);
    }
}
