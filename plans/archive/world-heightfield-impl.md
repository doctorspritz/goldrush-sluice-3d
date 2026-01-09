# Implementation Plan: Unified World Heightfield System

This plan is for Codex to implement. Completely decoupled from GPU particle work.

**Branch**: Create new branch `world-heightfield` from `master`
**Goal**: Unified terrain + water system with collapse, flow, and settling.

---

## Overview

Build the "world layer" that active particle zones sit on top of:

```
┌─────────────────────────────────────────┐
│           WATER SURFACE                 │  ← water_surface[x,z]
│  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    │
│         (suspended sediment)            │  ← suspended_sediment[x,z]
├─────────────────────────────────────────┤
│  ▓▓▓▓▓▓▓▓▓  SEDIMENT LAYER  ▓▓▓▓▓▓▓▓▓  │  ← terrain_sediment[x,z]
│  ▓▓▓▓▓▓▓▓▓  (deposited, grows) ▓▓▓▓▓▓  │     (grows from settling)
├─────────────────────────────────────────┤
│  ████████████  BASE TERRAIN  ███████████│  ← terrain_base[x,z]
│  ████████████  (bedrock/original) ██████│     (only changes via dig)
└─────────────────────────────────────────┘
```

---

## Phase 1: Extract Collapse Logic to sim3d

### 1.1 Create world.rs module

**Create**: `crates/sim3d/src/world.rs`

```rust
//! Unified world simulation: terrain + water + sediment settling.
//!
//! This is the "background" simulation for areas outside active particle zones.
//! Everything is heightfield-based for performance.

use glam::{Vec2, Vec3};

/// World simulation parameters
#[derive(Clone, Debug)]
pub struct WorldParams {
    /// Angle of repose for dry material (radians)
    pub angle_of_repose: f32,
    /// Fraction of excess height transferred per collapse step
    pub collapse_transfer_rate: f32,
    /// Maximum fraction of cell height that can flow out per step
    pub collapse_max_outflow: f32,
    /// Gravity (m/s²)
    pub gravity: f32,
    /// Water flow damping (0-1, higher = more damping)
    pub water_damping: f32,
    /// Sediment settling velocity (m/s)
    pub settling_velocity: f32,
    /// Bed porosity (0-1, fraction that is void space)
    pub bed_porosity: f32,
}

impl Default for WorldParams {
    fn default() -> Self {
        Self {
            angle_of_repose: 35.0_f32.to_radians(),
            collapse_transfer_rate: 0.35,
            collapse_max_outflow: 0.5,
            gravity: 9.81,
            water_damping: 0.02,
            settling_velocity: 0.01,
            bed_porosity: 0.4,
        }
    }
}

/// Material types for terrain
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TerrainMaterial {
    #[default]
    Dirt,
    Gravel,
    Sand,
    Clay,
    Bedrock,
}

impl TerrainMaterial {
    /// Angle of repose for this material (radians)
    pub fn angle_of_repose(self) -> f32 {
        match self {
            TerrainMaterial::Dirt => 35.0_f32.to_radians(),
            TerrainMaterial::Gravel => 38.0_f32.to_radians(),
            TerrainMaterial::Sand => 32.0_f32.to_radians(),
            TerrainMaterial::Clay => 45.0_f32.to_radians(),
            TerrainMaterial::Bedrock => 90.0_f32.to_radians(), // Doesn't collapse
        }
    }

    /// Density relative to water
    pub fn density(self) -> f32 {
        match self {
            TerrainMaterial::Dirt => 1.8,
            TerrainMaterial::Gravel => 2.0,
            TerrainMaterial::Sand => 1.6,
            TerrainMaterial::Clay => 1.9,
            TerrainMaterial::Bedrock => 2.5,
        }
    }
}

/// Unified world state
#[derive(Clone, Debug)]
pub struct World {
    pub width: usize,
    pub depth: usize,
    pub cell_size: f32,

    // Terrain layers
    /// Base terrain height (bedrock, original ground) - only changes via excavation
    pub terrain_base: Vec<f32>,
    /// Sediment layer height (deposited material) - grows from settling
    pub terrain_sediment: Vec<f32>,
    /// Material type per cell (affects angle of repose)
    pub terrain_material: Vec<TerrainMaterial>,

    // Water layer
    /// Water surface height (absolute, not depth)
    pub water_surface: Vec<f32>,
    /// Water flow velocity X component (at cell faces)
    pub water_flow_x: Vec<f32>,
    /// Water flow velocity Z component (at cell faces)
    pub water_flow_z: Vec<f32>,
    /// Suspended sediment concentration (volume fraction 0-1)
    pub suspended_sediment: Vec<f32>,

    // Working buffers (avoid allocation in update loop)
    collapse_deltas: Vec<f32>,

    // Parameters
    pub params: WorldParams,
}

impl World {
    /// Create a new world with flat terrain
    pub fn new(width: usize, depth: usize, cell_size: f32, initial_height: f32) -> Self {
        let cell_count = width * depth;
        let flow_x_count = (width + 1) * depth;
        let flow_z_count = width * (depth + 1);

        Self {
            width,
            depth,
            cell_size,
            terrain_base: vec![initial_height; cell_count],
            terrain_sediment: vec![0.0; cell_count],
            terrain_material: vec![TerrainMaterial::Dirt; cell_count],
            water_surface: vec![0.0; cell_count], // No water initially
            water_flow_x: vec![0.0; flow_x_count],
            water_flow_z: vec![0.0; flow_z_count],
            suspended_sediment: vec![0.0; cell_count],
            collapse_deltas: vec![0.0; cell_count],
            params: WorldParams::default(),
        }
    }

    /// Cell index from (x, z) coordinates
    #[inline]
    pub fn idx(&self, x: usize, z: usize) -> usize {
        z * self.width + x
    }

    /// Flow X index (faces between cells in X direction)
    #[inline]
    fn flow_x_idx(&self, x: usize, z: usize) -> usize {
        z * (self.width + 1) + x
    }

    /// Flow Z index (faces between cells in Z direction)
    #[inline]
    fn flow_z_idx(&self, x: usize, z: usize) -> usize {
        z * self.width + x
    }

    /// Total ground height (base + sediment)
    #[inline]
    pub fn ground_height(&self, x: usize, z: usize) -> f32 {
        let idx = self.idx(x, z);
        self.terrain_base[idx] + self.terrain_sediment[idx]
    }

    /// Water depth at cell
    #[inline]
    pub fn water_depth(&self, x: usize, z: usize) -> f32 {
        let idx = self.idx(x, z);
        (self.water_surface[idx] - self.ground_height(x, z)).max(0.0)
    }

    /// World bounds
    pub fn world_size(&self) -> Vec3 {
        Vec3::new(
            self.width as f32 * self.cell_size,
            100.0, // Arbitrary max height
            self.depth as f32 * self.cell_size,
        )
    }

    /// Convert world position to cell coordinates
    pub fn world_to_cell(&self, pos: Vec3) -> Option<(usize, usize)> {
        let x = (pos.x / self.cell_size) as i32;
        let z = (pos.z / self.cell_size) as i32;

        if x >= 0 && x < self.width as i32 && z >= 0 && z < self.depth as i32 {
            Some((x as usize, z as usize))
        } else {
            None
        }
    }

    /// Main update step
    pub fn update(&mut self, dt: f32) {
        self.update_terrain_collapse();
        self.update_water_flow(dt);
        self.update_sediment_settling(dt);
    }
}
```

### 1.2 Add terrain collapse (from dig_test.rs)

Add to `world.rs`:

```rust
impl World {
    /// Update terrain collapse based on angle of repose
    /// Returns true if any material moved
    pub fn update_terrain_collapse(&mut self) -> bool {
        let width = self.width;
        let depth = self.depth;
        let cell_size = self.cell_size;
        let transfer_rate = self.params.collapse_transfer_rate;
        let max_outflow = self.params.collapse_max_outflow;

        // Clear deltas
        self.collapse_deltas.fill(0.0);

        // Neighbor offsets: (dx, dz, distance)
        let neighbors: [(i32, i32, f32); 8] = [
            (1, 0, cell_size),
            (-1, 0, cell_size),
            (0, 1, cell_size),
            (0, -1, cell_size),
            (1, 1, cell_size * std::f32::consts::SQRT_2),
            (1, -1, cell_size * std::f32::consts::SQRT_2),
            (-1, 1, cell_size * std::f32::consts::SQRT_2),
            (-1, -1, cell_size * std::f32::consts::SQRT_2),
        ];

        // Calculate transfers
        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);
                let h = self.ground_height(x, z);

                // Get material's angle of repose
                let material = self.terrain_material[idx];
                if material == TerrainMaterial::Bedrock {
                    continue; // Bedrock doesn't collapse
                }
                let angle_tan = material.angle_of_repose().tan();

                let mut neighbor_transfers: [(usize, f32); 8] = [(0, 0.0); 8];
                let mut transfer_count = 0;
                let mut total_out = 0.0_f32;

                for (dx, dz, dist) in neighbors.iter() {
                    let nx = x as i32 + dx;
                    let nz = z as i32 + dz;

                    if nx < 0 || nz < 0 || nx >= width as i32 || nz >= depth as i32 {
                        continue;
                    }

                    let nidx = self.idx(nx as usize, nz as usize);
                    let nh = self.ground_height(nx as usize, nz as usize);
                    let diff = h - nh;
                    let max_diff = angle_tan * dist;

                    if diff > max_diff {
                        let transfer = transfer_rate * (diff - max_diff);
                        neighbor_transfers[transfer_count] = (nidx, transfer);
                        transfer_count += 1;
                        total_out += transfer;
                    }
                }

                if total_out <= 0.0 {
                    continue;
                }

                // Limit outflow to prevent negative heights
                let sediment_available = self.terrain_sediment[idx];
                let max_out = sediment_available * max_outflow;
                let scale = if total_out > max_out && total_out > 0.0 {
                    max_out / total_out
                } else {
                    1.0
                };

                // Apply scaled transfers
                for i in 0..transfer_count {
                    let (nidx, transfer) = neighbor_transfers[i];
                    let scaled_transfer = transfer * scale;
                    self.collapse_deltas[idx] -= scaled_transfer;
                    self.collapse_deltas[nidx] += scaled_transfer;
                }
            }
        }

        // Apply deltas to sediment layer
        let mut changed = false;
        for idx in 0..self.terrain_sediment.len() {
            if self.collapse_deltas[idx].abs() > 1e-6 {
                self.terrain_sediment[idx] =
                    (self.terrain_sediment[idx] + self.collapse_deltas[idx]).max(0.0);
                changed = true;
            }
        }

        changed
    }
}
```

### 1.3 Add excavation

Add to `world.rs`:

```rust
/// Information about material to spawn as particles
#[derive(Clone, Debug)]
pub struct ExcavationResult {
    pub position: Vec3,
    pub volume: f32,
    pub material: TerrainMaterial,
}

impl World {
    /// Excavate terrain at world position
    /// Returns list of (position, volume, material) for particle spawning
    pub fn excavate(
        &mut self,
        world_pos: Vec3,
        radius: f32,
        dig_depth: f32,
    ) -> Vec<ExcavationResult> {
        let mut results = Vec::new();

        let cx = (world_pos.x / self.cell_size) as i32;
        let cz = (world_pos.z / self.cell_size) as i32;
        let r_cells = (radius / self.cell_size).ceil() as i32;
        let r_sq = (radius / self.cell_size).powi(2);

        for dz in -r_cells..=r_cells {
            for dx in -r_cells..=r_cells {
                let dist_sq = (dx * dx + dz * dz) as f32;
                if dist_sq > r_sq {
                    continue;
                }

                let x = cx + dx;
                let z = cz + dz;

                if x < 0 || z < 0 || x >= self.width as i32 || z >= self.depth as i32 {
                    continue;
                }

                let x = x as usize;
                let z = z as usize;
                let idx = self.idx(x, z);

                // First dig from sediment layer
                let sed = self.terrain_sediment[idx];
                let base = self.terrain_base[idx];
                let material = self.terrain_material[idx];

                let mut remaining_dig = dig_depth;
                let mut total_dug = 0.0;

                // Dig sediment first
                if sed > 0.0 && remaining_dig > 0.0 {
                    let dug = remaining_dig.min(sed);
                    self.terrain_sediment[idx] -= dug;
                    remaining_dig -= dug;
                    total_dug += dug;
                }

                // Then dig base terrain (if not bedrock)
                if remaining_dig > 0.0 && material != TerrainMaterial::Bedrock {
                    let dug = remaining_dig.min(base);
                    self.terrain_base[idx] -= dug;
                    total_dug += dug;
                }

                if total_dug > 0.0 {
                    let cell_area = self.cell_size * self.cell_size;
                    let volume = total_dug * cell_area;

                    results.push(ExcavationResult {
                        position: Vec3::new(
                            (x as f32 + 0.5) * self.cell_size,
                            self.ground_height(x, z) + 0.1,
                            (z as f32 + 0.5) * self.cell_size,
                        ),
                        volume,
                        material,
                    });
                }
            }
        }

        results
    }

    /// Add material to terrain (building berms, dumping)
    pub fn add_material(
        &mut self,
        world_pos: Vec3,
        radius: f32,
        height: f32,
        material: TerrainMaterial,
    ) {
        let cx = (world_pos.x / self.cell_size) as i32;
        let cz = (world_pos.z / self.cell_size) as i32;
        let r_cells = (radius / self.cell_size).ceil() as i32;
        let r_sq = (radius / self.cell_size).powi(2);

        for dz in -r_cells..=r_cells {
            for dx in -r_cells..=r_cells {
                let dist_sq = (dx * dx + dz * dz) as f32;
                if dist_sq > r_sq {
                    continue;
                }

                let x = cx + dx;
                let z = cz + dz;

                if x < 0 || z < 0 || x >= self.width as i32 || z >= self.depth as i32 {
                    continue;
                }

                let idx = self.idx(x as usize, z as usize);

                // Add to sediment layer
                self.terrain_sediment[idx] += height;
                self.terrain_material[idx] = material;
            }
        }
    }
}
```

### 1.4 Update lib.rs

**Modify**: `crates/sim3d/src/lib.rs`

Add:

```rust
mod world;
pub use world::{World, WorldParams, TerrainMaterial, ExcavationResult};
```

---

## Phase 2: Water Flow (Shallow Water)

### 2.1 Add water flow update

Add to `world.rs`:

```rust
impl World {
    /// Update water flow using simplified shallow water equations
    pub fn update_water_flow(&mut self, dt: f32) {
        let g = self.params.gravity;
        let damping = 1.0 - self.params.water_damping;
        let width = self.width;
        let depth = self.depth;
        let cell_size = self.cell_size;

        // Step 1: Update flow velocities based on water surface gradient

        // X-direction flow (at cell faces)
        for z in 0..depth {
            for x in 1..width {
                let flow_idx = self.flow_x_idx(x, z);
                let idx_l = self.idx(x - 1, z);
                let idx_r = self.idx(x, z);

                let h_l = self.water_surface[idx_l];
                let h_r = self.water_surface[idx_r];

                // Only flow if there's water
                let depth_l = self.water_depth(x - 1, z);
                let depth_r = self.water_depth(x, z);

                if depth_l < 0.001 && depth_r < 0.001 {
                    self.water_flow_x[flow_idx] = 0.0;
                    continue;
                }

                // Accelerate based on surface gradient
                let gradient = (h_l - h_r) / cell_size;
                self.water_flow_x[flow_idx] += g * gradient * dt;
                self.water_flow_x[flow_idx] *= damping;

                // Limit flow to available water
                let avg_depth = (depth_l + depth_r) * 0.5;
                let max_flow = avg_depth * cell_size / dt * 0.25;
                self.water_flow_x[flow_idx] = self.water_flow_x[flow_idx].clamp(-max_flow, max_flow);
            }

            // Boundary: no flow at edges
            self.water_flow_x[self.flow_x_idx(0, z)] = 0.0;
            self.water_flow_x[self.flow_x_idx(width, z)] = 0.0;
        }

        // Z-direction flow
        for z in 1..depth {
            for x in 0..width {
                let flow_idx = self.flow_z_idx(x, z);
                let idx_b = self.idx(x, z - 1);
                let idx_f = self.idx(x, z);

                let h_b = self.water_surface[idx_b];
                let h_f = self.water_surface[idx_f];

                let depth_b = self.water_depth(x, z - 1);
                let depth_f = self.water_depth(x, z);

                if depth_b < 0.001 && depth_f < 0.001 {
                    self.water_flow_z[flow_idx] = 0.0;
                    continue;
                }

                let gradient = (h_b - h_f) / cell_size;
                self.water_flow_z[flow_idx] += g * gradient * dt;
                self.water_flow_z[flow_idx] *= damping;

                let avg_depth = (depth_b + depth_f) * 0.5;
                let max_flow = avg_depth * cell_size / dt * 0.25;
                self.water_flow_z[flow_idx] = self.water_flow_z[flow_idx].clamp(-max_flow, max_flow);
            }
        }

        // Boundary: no flow at edges
        for x in 0..width {
            self.water_flow_z[self.flow_z_idx(x, 0)] = 0.0;
            self.water_flow_z[x + depth * width] = 0.0; // z = depth boundary
        }

        // Step 2: Update water surface based on flow divergence
        let cell_area = cell_size * cell_size;

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);

                // Net flow into cell
                let flow_in_x = self.water_flow_x[self.flow_x_idx(x, z)]
                    - self.water_flow_x[self.flow_x_idx(x + 1, z)];

                let flow_in_z = if z > 0 {
                    self.water_flow_z[self.flow_z_idx(x, z - 1)]
                } else {
                    0.0
                } - self.water_flow_z[self.flow_z_idx(x, z)];

                let volume_change = (flow_in_x + flow_in_z) * cell_size * dt;
                let height_change = volume_change / cell_area;

                self.water_surface[idx] += height_change;

                // Water can't go below ground
                let ground = self.ground_height(x, z);
                self.water_surface[idx] = self.water_surface[idx].max(ground);
            }
        }

        // Step 3: Advect suspended sediment with water flow
        self.advect_suspended_sediment(dt);
    }

    /// Advect suspended sediment with water flow (simple upwind)
    fn advect_suspended_sediment(&mut self, dt: f32) {
        let width = self.width;
        let depth = self.depth;
        let cell_size = self.cell_size;

        // Simple first-order upwind advection
        let mut new_sediment = self.suspended_sediment.clone();

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);
                let water_depth = self.water_depth(x, z);

                if water_depth < 0.001 {
                    new_sediment[idx] = 0.0;
                    continue;
                }

                // Get flow velocities at cell center (average of faces)
                let vx = (self.water_flow_x[self.flow_x_idx(x, z)]
                    + self.water_flow_x[self.flow_x_idx(x + 1, z)]) * 0.5;
                let vz = (self.water_flow_z[self.flow_z_idx(x, z)]
                    + if z + 1 < depth {
                        self.water_flow_z[self.flow_z_idx(x, z + 1)]
                    } else {
                        0.0
                    }) * 0.5;

                // Upwind: look at where flow is coming from
                let mut c = self.suspended_sediment[idx];

                if vx > 0.0 && x > 0 {
                    let c_upwind = self.suspended_sediment[self.idx(x - 1, z)];
                    c -= vx * dt / cell_size * (c - c_upwind);
                } else if vx < 0.0 && x + 1 < width {
                    let c_upwind = self.suspended_sediment[self.idx(x + 1, z)];
                    c -= vx * dt / cell_size * (c_upwind - c);
                }

                if vz > 0.0 && z > 0 {
                    let c_upwind = self.suspended_sediment[self.idx(x, z - 1)];
                    c -= vz * dt / cell_size * (c - c_upwind);
                } else if vz < 0.0 && z + 1 < depth {
                    let c_upwind = self.suspended_sediment[self.idx(x, z + 1)];
                    c -= vz * dt / cell_size * (c_upwind - c);
                }

                new_sediment[idx] = c.max(0.0);
            }
        }

        self.suspended_sediment = new_sediment;
    }
}
```

### 2.2 Add water input/output

Add to `world.rs`:

```rust
impl World {
    /// Add water at a position
    pub fn add_water(&mut self, world_pos: Vec3, volume: f32) {
        if let Some((x, z)) = self.world_to_cell(world_pos) {
            let idx = self.idx(x, z);
            let cell_area = self.cell_size * self.cell_size;
            let height_add = volume / cell_area;

            // Ensure water surface is at least at ground level
            let ground = self.ground_height(x, z);
            if self.water_surface[idx] < ground {
                self.water_surface[idx] = ground;
            }

            self.water_surface[idx] += height_add;
        }
    }

    /// Add sediment-laden water (from active zone outflow)
    pub fn add_sediment_water(
        &mut self,
        world_pos: Vec3,
        water_volume: f32,
        sediment_volume: f32,
    ) {
        if let Some((x, z)) = self.world_to_cell(world_pos) {
            let idx = self.idx(x, z);
            let cell_area = self.cell_size * self.cell_size;

            // Add water
            let height_add = water_volume / cell_area;
            let ground = self.ground_height(x, z);
            if self.water_surface[idx] < ground {
                self.water_surface[idx] = ground;
            }
            self.water_surface[idx] += height_add;

            // Add suspended sediment
            let water_depth = self.water_depth(x, z);
            if water_depth > 0.001 {
                let conc_add = sediment_volume / (cell_area * water_depth);
                self.suspended_sediment[idx] = (self.suspended_sediment[idx] + conc_add).min(0.5);
            }
        }
    }

    /// Remove water at a position (pumping, drainage)
    /// Returns actual volume removed
    pub fn remove_water(&mut self, world_pos: Vec3, max_volume: f32) -> f32 {
        if let Some((x, z)) = self.world_to_cell(world_pos) {
            let idx = self.idx(x, z);
            let cell_area = self.cell_size * self.cell_size;

            let depth = self.water_depth(x, z);
            let available = depth * cell_area;
            let removed = available.min(max_volume);

            let height_remove = removed / cell_area;
            self.water_surface[idx] -= height_remove;

            removed
        } else {
            0.0
        }
    }

    /// Get total water volume in world
    pub fn total_water_volume(&self) -> f32 {
        let cell_area = self.cell_size * self.cell_size;
        let mut total = 0.0;

        for z in 0..self.depth {
            for x in 0..self.width {
                total += self.water_depth(x, z) * cell_area;
            }
        }

        total
    }
}
```

---

## Phase 3: Sediment Settling

### 3.1 Add settling update

Add to `world.rs`:

```rust
impl World {
    /// Update sediment settling (suspended → terrain_sediment)
    pub fn update_sediment_settling(&mut self, dt: f32) {
        let settling_velocity = self.params.settling_velocity;
        let bed_porosity = self.params.bed_porosity;
        let cell_area = self.cell_size * self.cell_size;

        for z in 0..self.depth {
            for x in 0..self.width {
                let idx = self.idx(x, z);
                let depth = self.water_depth(x, z);

                if depth < 0.001 {
                    // No water, no settling
                    self.suspended_sediment[idx] = 0.0;
                    continue;
                }

                let conc = self.suspended_sediment[idx];
                if conc < 0.0001 {
                    continue;
                }

                // Volume of sediment that settles this timestep
                // settling_height is how far sediment falls in dt
                let settling_height = settling_velocity * dt;
                let settled_fraction = (settling_height / depth).min(1.0);

                // Volume settling out of water column
                let settled_volume = conc * cell_area * depth * settled_fraction;

                // Remove from suspension
                self.suspended_sediment[idx] *= 1.0 - settled_fraction;

                // Add to terrain sediment layer (accounting for porosity)
                let solid_fraction = 1.0 - bed_porosity;
                let bed_height_increase = settled_volume / (cell_area * solid_fraction);
                self.terrain_sediment[idx] += bed_height_increase;

                // If sediment builds up to water surface, cap it
                let max_sediment = self.water_surface[idx] - self.terrain_base[idx];
                if self.terrain_sediment[idx] > max_sediment {
                    self.terrain_sediment[idx] = max_sediment.max(0.0);
                }
            }
        }
    }

    /// Get total deposited sediment volume
    pub fn total_sediment_volume(&self) -> f32 {
        let cell_area = self.cell_size * self.cell_size;
        self.terrain_sediment.iter().sum::<f32>() * cell_area
    }
}
```

---

## Phase 4: Rendering Helpers

### 4.1 Add mesh generation helpers

Add to `world.rs`:

```rust
impl World {
    /// Get vertex data for terrain mesh rendering
    /// Returns (positions, colors) where each vertex has position and color
    pub fn terrain_vertices(&self) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let mut positions = Vec::with_capacity(self.width * self.depth);
        let mut colors = Vec::with_capacity(self.width * self.depth);

        for z in 0..self.depth {
            for x in 0..self.width {
                let idx = self.idx(x, z);
                let height = self.ground_height(x, z);
                let sediment = self.terrain_sediment[idx];

                positions.push([
                    (x as f32 + 0.5) * self.cell_size,
                    height,
                    (z as f32 + 0.5) * self.cell_size,
                ]);

                // Color based on sediment depth
                let sediment_ratio = (sediment / 2.0).min(1.0); // 2m = full color
                let base_color = match self.terrain_material[idx] {
                    TerrainMaterial::Dirt => [0.4, 0.3, 0.2],
                    TerrainMaterial::Gravel => [0.5, 0.5, 0.5],
                    TerrainMaterial::Sand => [0.8, 0.7, 0.5],
                    TerrainMaterial::Clay => [0.6, 0.4, 0.3],
                    TerrainMaterial::Bedrock => [0.3, 0.3, 0.35],
                };

                // Sediment is lighter brown
                let sediment_color = [0.6, 0.5, 0.4];

                colors.push([
                    base_color[0] * (1.0 - sediment_ratio) + sediment_color[0] * sediment_ratio,
                    base_color[1] * (1.0 - sediment_ratio) + sediment_color[1] * sediment_ratio,
                    base_color[2] * (1.0 - sediment_ratio) + sediment_color[2] * sediment_ratio,
                ]);
            }
        }

        (positions, colors)
    }

    /// Get vertex data for water surface mesh
    /// Returns (positions, colors) - only for cells with water
    pub fn water_vertices(&self) -> (Vec<[f32; 3]>, Vec<[f32; 4]>) {
        let mut positions = Vec::new();
        let mut colors = Vec::new();

        for z in 0..self.depth {
            for x in 0..self.width {
                let depth = self.water_depth(x, z);
                if depth < 0.01 {
                    continue;
                }

                let idx = self.idx(x, z);
                let height = self.water_surface[idx];
                let turbidity = self.suspended_sediment[idx];

                positions.push([
                    (x as f32 + 0.5) * self.cell_size,
                    height,
                    (z as f32 + 0.5) * self.cell_size,
                ]);

                // Color: blue, brown with turbidity, more opaque when deeper
                let alpha = (depth / 2.0).min(0.8);
                let brown = turbidity.min(0.5) * 2.0; // 0.5 conc = full brown

                colors.push([
                    0.2 + brown * 0.4,  // R: more red when turbid
                    0.4 + brown * 0.2,  // G
                    0.8 - brown * 0.4,  // B: less blue when turbid
                    alpha,
                ]);
            }
        }

        (positions, colors)
    }
}
```

---

## Phase 5: Test Example

### 5.1 Create world_test.rs example

**Create**: `crates/game/examples/world_test.rs`

```rust
//! World Heightfield Test
//!
//! Tests the unified world system with terrain collapse, water flow, and settling.
//!
//! Controls:
//! - WASD: Move camera
//! - SPACE/SHIFT: Up/Down
//! - Right Mouse: Look
//! - Left Mouse: Dig terrain
//! - Ctrl + Left Mouse: Add material
//! - 1: Add water at cursor
//! - 2: Add muddy water at cursor
//! - R: Reset world
//! - ESC: Quit
//!
//! Run: cargo run --example world_test --release

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use sim3d::{World, WorldParams, TerrainMaterial};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const WORLD_WIDTH: usize = 100;
const WORLD_DEPTH: usize = 100;
const CELL_SIZE: f32 = 1.0;
const INITIAL_HEIGHT: f32 = 10.0;

const DIG_RADIUS: f32 = 3.0;
const DIG_DEPTH: f32 = 0.5;
const ADD_RADIUS: f32 = 3.0;
const ADD_HEIGHT: f32 = 0.5;
const WATER_ADD_VOLUME: f32 = 5.0;

const MOVE_SPEED: f32 = 20.0;
const MOUSE_SENSITIVITY: f32 = 0.003;

const STEPS_PER_FRAME: usize = 4;
const DT: f32 = 1.0 / 60.0 / STEPS_PER_FRAME as f32;

// ... (Camera, InputState, App structs similar to dig_test.rs)

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    world: World,
    camera: Camera,
    input: InputState,
    last_frame: Instant,
    window_size: (u32, u32),
}

impl App {
    fn new() -> Self {
        let mut world = World::new(WORLD_WIDTH, WORLD_DEPTH, CELL_SIZE, INITIAL_HEIGHT);

        // Create a depression for water to collect
        for z in 30..70 {
            for x in 30..70 {
                let idx = world.idx(x, z);
                let dist_x = (x as f32 - 50.0).abs();
                let dist_z = (z as f32 - 50.0).abs();
                let dist = (dist_x.max(dist_z)) / 20.0;
                world.terrain_base[idx] = INITIAL_HEIGHT - 5.0 * (1.0 - dist).max(0.0);
            }
        }

        // Add a ridge/berm
        for z in 68..72 {
            for x in 30..70 {
                let idx = world.idx(x, z);
                world.terrain_sediment[idx] = 3.0;
            }
        }

        Self {
            window: None,
            gpu: None,
            world,
            camera: Camera {
                position: Vec3::new(50.0, 30.0, 80.0),
                yaw: -1.57,
                pitch: -0.4,
                speed: MOVE_SPEED,
                sensitivity: MOUSE_SENSITIVITY,
            },
            input: InputState::default(),
            last_frame: Instant::now(),
            window_size: (1280, 720),
        }
    }

    fn update(&mut self, dt: f32) {
        // Multiple physics steps per frame for stability
        for _ in 0..STEPS_PER_FRAME {
            self.world.update(DT);
        }

        // Handle input
        if self.input.left_mouse {
            if let Some(hit) = self.raycast_terrain() {
                if self.input.ctrl {
                    self.world.add_material(hit, ADD_RADIUS, ADD_HEIGHT, TerrainMaterial::Gravel);
                } else {
                    self.world.excavate(hit, DIG_RADIUS, DIG_DEPTH);
                }
            }
        }
    }

    fn add_water_at_cursor(&mut self) {
        if let Some(hit) = self.raycast_terrain() {
            self.world.add_water(hit, WATER_ADD_VOLUME);
        }
    }

    fn add_muddy_water_at_cursor(&mut self) {
        if let Some(hit) = self.raycast_terrain() {
            self.world.add_sediment_water(hit, WATER_ADD_VOLUME, WATER_ADD_VOLUME * 0.1);
        }
    }

    fn raycast_terrain(&self) -> Option<Vec3> {
        // Simple raycast against terrain heightfield
        let ray_dir = self.camera.forward();
        let ray_origin = self.camera.position;

        let step = 0.5;
        let max_dist = 200.0;

        let mut t = 0.0;
        while t < max_dist {
            let p = ray_origin + ray_dir * t;

            if let Some((x, z)) = self.world.world_to_cell(p) {
                let ground = self.world.ground_height(x, z);
                if p.y <= ground {
                    return Some(p);
                }
            }

            t += step;
        }

        None
    }
}

// ... (rest of rendering code similar to dig_test.rs, but renders both terrain and water)

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
```

**Note**: The full rendering code follows the pattern in `dig_test.rs`. Key additions:
- Render water surface as semi-transparent blue mesh
- Show turbidity via water color
- Display stats: water volume, sediment deposited

---

## Phase 6: Tests

### 6.1 Unit tests

**Add to**: `crates/sim3d/src/world.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_creation() {
        let world = World::new(10, 10, 1.0, 5.0);
        assert_eq!(world.width, 10);
        assert_eq!(world.depth, 10);
        assert!((world.ground_height(5, 5) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_excavation() {
        let mut world = World::new(10, 10, 1.0, 5.0);
        let results = world.excavate(Vec3::new(5.0, 0.0, 5.0), 1.0, 1.0);

        assert!(!results.is_empty());
        assert!(world.ground_height(5, 5) < 5.0);
    }

    #[test]
    fn test_terrain_collapse() {
        let mut world = World::new(10, 10, 1.0, 0.0);

        // Create a spike
        world.terrain_sediment[world.idx(5, 5)] = 10.0;

        // Run collapse
        for _ in 0..100 {
            world.update_terrain_collapse();
        }

        // Spike should have spread out
        let center_height = world.terrain_sediment[world.idx(5, 5)];
        let neighbor_height = world.terrain_sediment[world.idx(5, 6)];

        assert!(center_height < 10.0); // Center lowered
        assert!(neighbor_height > 0.0); // Neighbors raised
    }

    #[test]
    fn test_water_leveling() {
        let mut world = World::new(10, 10, 1.0, 0.0);

        // Add water to one corner
        world.add_water(Vec3::new(1.0, 0.0, 1.0), 100.0);

        let initial_total = world.total_water_volume();

        // Run flow
        for _ in 0..1000 {
            world.update_water_flow(0.016);
        }

        // Water should spread and level
        let final_total = world.total_water_volume();

        // Mass conserved (within tolerance)
        assert!((initial_total - final_total).abs() < 1.0);

        // Water should be roughly level (check corners)
        let depth_00 = world.water_depth(0, 0);
        let depth_99 = world.water_depth(9, 9);
        assert!((depth_00 - depth_99).abs() < 0.5);
    }

    #[test]
    fn test_sediment_settling() {
        let mut world = World::new(10, 10, 1.0, 0.0);

        // Add water with sediment
        world.add_sediment_water(Vec3::new(5.0, 0.0, 5.0), 10.0, 1.0);

        let initial_suspended = world.suspended_sediment[world.idx(5, 5)];
        assert!(initial_suspended > 0.0);

        // Run settling
        for _ in 0..1000 {
            world.update_sediment_settling(0.016);
        }

        // Sediment should have settled
        let final_suspended = world.suspended_sediment[world.idx(5, 5)];
        let final_bed = world.terrain_sediment[world.idx(5, 5)];

        assert!(final_suspended < initial_suspended);
        assert!(final_bed > 0.0);
    }
}
```

---

## File Summary

| File | Action |
|------|--------|
| `crates/sim3d/src/world.rs` | CREATE |
| `crates/sim3d/src/lib.rs` | MODIFY (add export) |
| `crates/game/examples/world_test.rs` | CREATE |

---

## Checklist

- [ ] Phase 1: Create `world.rs` with World struct
- [ ] Phase 1: Add terrain collapse logic
- [ ] Phase 1: Add excavation function
- [ ] Phase 1: Add material addition function
- [ ] Phase 1: Update `lib.rs` exports
- [ ] Phase 2: Add water flow update (shallow water)
- [ ] Phase 2: Add sediment advection
- [ ] Phase 2: Add water input/output functions
- [ ] Phase 3: Add sediment settling
- [ ] Phase 4: Add rendering helpers
- [ ] Phase 5: Create `world_test.rs` example
- [ ] Phase 6: Add unit tests
- [ ] Test: Terrain collapse works
- [ ] Test: Water levels out
- [ ] Test: Sediment settles and builds bed
- [ ] Test: Building berms contains water

---

## Integration Points (Future)

After this is working, integration with active particle zones:

```rust
// Active zone drains to world
world.add_sediment_water(
    active_zone.outlet_position,
    active_zone.water_outflow_volume(),
    active_zone.sediment_outflow_volume(),
);

// World water enters active zone
let inflow = world.remove_water(
    active_zone.inlet_position,
    active_zone.max_inflow_volume(),
);
active_zone.add_water_inflow(inflow);

// Excavating world spawns particles in active zone
let excavated = world.excavate(dig_pos, radius, depth);
for result in excavated {
    active_zone.spawn_particles(result.position, result.volume, result.material);
}
```

These integration points are NOT part of this plan - just showing how it connects.
