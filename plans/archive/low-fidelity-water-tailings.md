# Plan: Low-Fidelity Water & Tailings System

## Overview

Large-scale water simulation for areas outside active particle zones. Cheap heightfield-based approach that:
- Fills depressions and levels out
- Transports suspended sediment
- Settles sediment to grow bed over time (tailings fill up!)
- Integrates with active FLIP/APIC zones at boundaries

## Goals

1. **Tailings ponds** that fill with sediment over time
2. **Flooded mine pits** with realistic water levels
3. **Rivers/channels** for water supply and drainage
4. **Seamless transitions** to/from particle simulation

## Data Structures

### WaterBody (Generic heightfield water)

```rust
// crates/sim3d/src/water_body.rs

/// Low-fidelity water body using shallow water equations
#[derive(Clone, Debug)]
pub struct WaterBody {
    pub width: usize,
    pub depth: usize,
    pub cell_size: f32,

    /// Reference to terrain heightfield (borrowed or index)
    pub terrain_offset: Vec3,  // World position of (0,0) corner

    /// Water surface height per (x,z) cell
    pub surface_height: Vec<f32>,

    /// Flow velocities (cell faces)
    pub flow_x: Vec<f32>,  // Flow in +X direction, size: (width+1) * depth
    pub flow_z: Vec<f32>,  // Flow in +Z direction, size: width * (depth+1)

    /// Boundary conditions
    pub inflow_rate: f32,      // m³/s from particle zone
    pub outflow_positions: Vec<(usize, usize)>,  // Spillway/overflow points
}
```

### TailingsPond (Specialization with sediment)

```rust
// crates/sim3d/src/tailings.rs

/// Tailings pond with sediment settling
#[derive(Clone, Debug)]
pub struct TailingsPond {
    pub width: usize,
    pub depth: usize,
    pub cell_size: f32,
    pub world_offset: Vec3,

    // Three-layer heightfield system
    pub terrain_height: Vec<f32>,    // Base ground (static)
    pub sediment_height: Vec<f32>,   // Settled tailings (grows!)
    pub water_surface: Vec<f32>,     // Water level

    // Sediment in water column (per cell, depth-averaged)
    pub suspended_conc: Vec<f32>,    // Volume fraction (0.0 - 1.0)

    // Flow
    pub flow_x: Vec<f32>,
    pub flow_z: Vec<f32>,

    // Parameters
    pub params: TailingsParams,

    // Stats
    pub total_sediment_deposited: f32,  // m³, cumulative
    pub capacity_fraction: f32,          // 0.0 = empty, 1.0 = full
}

#[derive(Clone, Debug)]
pub struct TailingsParams {
    pub settling_velocity: f32,     // m/s, ~0.01 for fine sand
    pub bed_porosity: f32,          // ~0.4
    pub max_suspended_conc: f32,    // ~0.3 volume fraction
    pub inflow_sediment_conc: f32,  // From sluice, ~0.1
    pub overflow_height: f32,       // Dam crest height
}

impl Default for TailingsParams {
    fn default() -> Self {
        Self {
            settling_velocity: 0.01,
            bed_porosity: 0.4,
            max_suspended_conc: 0.3,
            inflow_sediment_conc: 0.1,
            overflow_height: 10.0,
        }
    }
}
```

---

## Implementation Phases

### Phase 1: Basic Water Heightfield

**Goal**: Water that flows and levels out.

#### 1.1 Create water_body.rs

```rust
// crates/sim3d/src/water_body.rs

use glam::Vec3;

pub struct WaterBody {
    pub width: usize,
    pub depth: usize,
    pub cell_size: f32,
    pub world_offset: Vec3,
    pub surface_height: Vec<f32>,
    pub flow_x: Vec<f32>,
    pub flow_z: Vec<f32>,
}

impl WaterBody {
    pub fn new(
        width: usize,
        depth: usize,
        cell_size: f32,
        world_offset: Vec3,
        initial_height: f32,
    ) -> Self {
        Self {
            width,
            depth,
            cell_size,
            world_offset,
            surface_height: vec![initial_height; width * depth],
            flow_x: vec![0.0; (width + 1) * depth],
            flow_z: vec![0.0; width * (depth + 1)],
        }
    }

    fn idx(&self, x: usize, z: usize) -> usize {
        z * self.width + x
    }

    fn flow_x_idx(&self, x: usize, z: usize) -> usize {
        z * (self.width + 1) + x
    }

    fn flow_z_idx(&self, x: usize, z: usize) -> usize {
        z * self.width + x
    }

    /// Update water flow and surface (shallow water equations, simplified)
    pub fn update(&mut self, terrain: &[f32], dt: f32) {
        let g = 9.81_f32;
        let damping = 0.98_f32;

        // Step 1: Update flow based on surface gradient
        // X-direction flow
        for z in 0..self.depth {
            for x in 0..=self.width {
                let idx = self.flow_x_idx(x, z);

                if x == 0 || x == self.width {
                    // Boundary: no flow
                    self.flow_x[idx] = 0.0;
                    continue;
                }

                let idx_l = self.idx(x - 1, z);
                let idx_r = self.idx(x, z);

                let h_l = self.surface_height[idx_l];
                let h_r = self.surface_height[idx_r];

                // Accelerate flow based on pressure gradient
                let gradient = (h_l - h_r) / self.cell_size;
                self.flow_x[idx] += g * gradient * dt;
                self.flow_x[idx] *= damping;

                // Limit flow to available water
                let depth_l = (h_l - terrain[idx_l]).max(0.0);
                let depth_r = (h_r - terrain[idx_r]).max(0.0);
                let max_flow = depth_l.min(depth_r) * self.cell_size / dt;
                self.flow_x[idx] = self.flow_x[idx].clamp(-max_flow, max_flow);
            }
        }

        // Z-direction flow (similar)
        for z in 0..=self.depth {
            for x in 0..self.width {
                let idx = self.flow_z_idx(x, z);

                if z == 0 || z == self.depth {
                    self.flow_z[idx] = 0.0;
                    continue;
                }

                let idx_b = self.idx(x, z - 1);
                let idx_f = self.idx(x, z);

                let h_b = self.surface_height[idx_b];
                let h_f = self.surface_height[idx_f];

                let gradient = (h_b - h_f) / self.cell_size;
                self.flow_z[idx] += g * gradient * dt;
                self.flow_z[idx] *= damping;

                let depth_b = (h_b - terrain[idx_b]).max(0.0);
                let depth_f = (h_f - terrain[idx_f]).max(0.0);
                let max_flow = depth_b.min(depth_f) * self.cell_size / dt;
                self.flow_z[idx] = self.flow_z[idx].clamp(-max_flow, max_flow);
            }
        }

        // Step 2: Update water heights based on flow divergence
        let cell_area = self.cell_size * self.cell_size;

        for z in 0..self.depth {
            for x in 0..self.width {
                let idx = self.idx(x, z);

                // Net flow into cell
                let flow_in_x = self.flow_x[self.flow_x_idx(x, z)]
                              - self.flow_x[self.flow_x_idx(x + 1, z)];
                let flow_in_z = self.flow_z[self.flow_z_idx(x, z)]
                              - self.flow_z[self.flow_z_idx(x, z + 1)];

                let volume_change = (flow_in_x + flow_in_z) * self.cell_size * dt;
                let height_change = volume_change / cell_area;

                self.surface_height[idx] += height_change;

                // Clamp to terrain (can't go below ground)
                self.surface_height[idx] = self.surface_height[idx].max(terrain[idx]);
            }
        }
    }

    /// Add water at a point (from particle zone outflow)
    pub fn add_water(&mut self, world_x: f32, world_z: f32, volume: f32) {
        let lx = ((world_x - self.world_offset.x) / self.cell_size) as usize;
        let lz = ((world_z - self.world_offset.z) / self.cell_size) as usize;

        if lx < self.width && lz < self.depth {
            let idx = self.idx(lx, lz);
            let height_add = volume / (self.cell_size * self.cell_size);
            self.surface_height[idx] += height_add;
        }
    }

    /// Get water depth at world position
    pub fn get_depth(&self, terrain: &[f32], world_x: f32, world_z: f32) -> f32 {
        let lx = ((world_x - self.world_offset.x) / self.cell_size) as usize;
        let lz = ((world_z - self.world_offset.z) / self.cell_size) as usize;

        if lx < self.width && lz < self.depth {
            let idx = self.idx(lx, lz);
            (self.surface_height[idx] - terrain[idx]).max(0.0)
        } else {
            0.0
        }
    }
}
```

#### 1.2 Add to sim3d lib.rs

```rust
mod water_body;
pub use water_body::WaterBody;
```

#### 1.3 Test: Basic water leveling

Create test that:
1. Creates water body with uneven surface
2. Runs update for N steps
3. Verifies surface becomes flat

---

### Phase 2: Tailings Pond with Settling

**Goal**: Sediment settles and builds up bed over time.

#### 2.1 Create tailings.rs

```rust
// crates/sim3d/src/tailings.rs

use glam::Vec3;

#[derive(Clone, Debug)]
pub struct TailingsPond {
    pub width: usize,
    pub depth: usize,
    pub cell_size: f32,
    pub world_offset: Vec3,

    pub terrain_height: Vec<f32>,
    pub sediment_height: Vec<f32>,
    pub water_surface: Vec<f32>,
    pub suspended_conc: Vec<f32>,

    pub flow_x: Vec<f32>,
    pub flow_z: Vec<f32>,

    pub params: TailingsParams,

    pub total_sediment_deposited: f32,
    pub overflow_sediment: f32,
}

#[derive(Clone, Debug)]
pub struct TailingsParams {
    pub settling_velocity: f32,
    pub bed_porosity: f32,
    pub max_suspended_conc: f32,
    pub overflow_height: f32,
    pub diffusion_rate: f32,
}

impl Default for TailingsParams {
    fn default() -> Self {
        Self {
            settling_velocity: 0.01,
            bed_porosity: 0.4,
            max_suspended_conc: 0.3,
            overflow_height: 10.0,
            diffusion_rate: 0.1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TailingsStatus {
    Normal,
    Warning,    // >50% full
    Critical,   // >75% full
    Overflow,   // >90% full, sediment spilling
}

impl TailingsPond {
    pub fn new(
        width: usize,
        depth: usize,
        cell_size: f32,
        world_offset: Vec3,
        terrain_height: Vec<f32>,
        initial_water_height: f32,
    ) -> Self {
        let cell_count = width * depth;

        Self {
            width,
            depth,
            cell_size,
            world_offset,
            terrain_height,
            sediment_height: vec![0.0; cell_count],
            water_surface: vec![initial_water_height; cell_count],
            suspended_conc: vec![0.0; cell_count],
            flow_x: vec![0.0; (width + 1) * depth],
            flow_z: vec![0.0; width * (depth + 1)],
            params: TailingsParams::default(),
            total_sediment_deposited: 0.0,
            overflow_sediment: 0.0,
        }
    }

    fn idx(&self, x: usize, z: usize) -> usize {
        z * self.width + x
    }

    /// Effective bed = terrain + settled sediment
    pub fn effective_bed(&self, idx: usize) -> f32 {
        self.terrain_height[idx] + self.sediment_height[idx]
    }

    /// Water depth at cell
    pub fn water_depth(&self, idx: usize) -> f32 {
        (self.water_surface[idx] - self.effective_bed(idx)).max(0.0)
    }

    /// Total remaining capacity (m³)
    pub fn capacity_remaining(&self) -> f32 {
        let cell_area = self.cell_size * self.cell_size;
        let mut total = 0.0;

        for idx in 0..self.terrain_height.len() {
            let space = self.params.overflow_height - self.effective_bed(idx);
            total += space.max(0.0) * cell_area;
        }

        total
    }

    /// Current fill fraction (0.0 = empty, 1.0 = full)
    pub fn fill_fraction(&self) -> f32 {
        let cell_area = self.cell_size * self.cell_size;
        let mut filled = 0.0;
        let mut total = 0.0;

        for idx in 0..self.terrain_height.len() {
            let base = self.terrain_height[idx];
            let sed = self.sediment_height[idx];
            let max_h = self.params.overflow_height;

            filled += sed * cell_area;
            total += (max_h - base).max(0.0) * cell_area;
        }

        if total > 0.0 { filled / total } else { 1.0 }
    }

    /// Check operational status
    pub fn status(&self) -> TailingsStatus {
        let fill = self.fill_fraction();
        match fill {
            f if f < 0.5 => TailingsStatus::Normal,
            f if f < 0.75 => TailingsStatus::Warning,
            f if f < 0.9 => TailingsStatus::Critical,
            _ => TailingsStatus::Overflow,
        }
    }

    /// Add sediment-laden water at inlet
    pub fn add_inflow(
        &mut self,
        world_x: f32,
        world_z: f32,
        water_volume: f32,
        sediment_volume: f32,
    ) {
        let lx = ((world_x - self.world_offset.x) / self.cell_size) as usize;
        let lz = ((world_z - self.world_offset.z) / self.cell_size) as usize;

        if lx >= self.width || lz >= self.depth {
            return;
        }

        let idx = self.idx(lx, lz);
        let cell_area = self.cell_size * self.cell_size;

        // Add water
        let water_height_add = water_volume / cell_area;
        self.water_surface[idx] += water_height_add;

        // Add suspended sediment
        let depth = self.water_depth(idx);
        if depth > 0.001 {
            let conc_add = sediment_volume / (cell_area * depth);
            self.suspended_conc[idx] = (self.suspended_conc[idx] + conc_add)
                .min(self.params.max_suspended_conc);
        }
    }

    /// Main update step
    pub fn update(&mut self, dt: f32) {
        self.update_flow(dt);
        self.update_settling(dt);
        self.update_diffusion(dt);
        self.handle_overflow();
    }

    fn update_flow(&mut self, dt: f32) {
        let g = 9.81_f32;
        let damping = 0.98_f32;

        // Update X flow
        for z in 0..self.depth {
            for x in 1..self.width {
                let flow_idx = z * (self.width + 1) + x;
                let idx_l = self.idx(x - 1, z);
                let idx_r = self.idx(x, z);

                let h_l = self.water_surface[idx_l];
                let h_r = self.water_surface[idx_r];

                let gradient = (h_l - h_r) / self.cell_size;
                self.flow_x[flow_idx] += g * gradient * dt;
                self.flow_x[flow_idx] *= damping;

                // Limit to available water
                let depth_l = self.water_depth(idx_l);
                let depth_r = self.water_depth(idx_r);
                let max_flow = depth_l.min(depth_r) * self.cell_size / dt * 0.25;
                self.flow_x[flow_idx] = self.flow_x[flow_idx].clamp(-max_flow, max_flow);
            }
        }

        // Update Z flow (similar)
        for z in 1..self.depth {
            for x in 0..self.width {
                let flow_idx = z * self.width + x;
                let idx_b = self.idx(x, z - 1);
                let idx_f = self.idx(x, z);

                let h_b = self.water_surface[idx_b];
                let h_f = self.water_surface[idx_f];

                let gradient = (h_b - h_f) / self.cell_size;
                self.flow_z[flow_idx] += g * gradient * dt;
                self.flow_z[flow_idx] *= damping;

                let depth_b = self.water_depth(idx_b);
                let depth_f = self.water_depth(idx_f);
                let max_flow = depth_b.min(depth_f) * self.cell_size / dt * 0.25;
                self.flow_z[flow_idx] = self.flow_z[flow_idx].clamp(-max_flow, max_flow);
            }
        }

        // Apply flow to water surface and advect sediment
        let cell_area = self.cell_size * self.cell_size;

        for z in 0..self.depth {
            for x in 0..self.width {
                let idx = self.idx(x, z);

                // Flow divergence
                let fx_l = self.flow_x[z * (self.width + 1) + x];
                let fx_r = self.flow_x[z * (self.width + 1) + x + 1];
                let fz_b = if z > 0 { self.flow_z[(z - 1) * self.width + x] } else { 0.0 };
                let fz_f = self.flow_z[z * self.width + x];

                let net_flow = (fx_l - fx_r + fz_b - fz_f) * self.cell_size * dt;
                let height_change = net_flow / cell_area;

                self.water_surface[idx] += height_change;
                self.water_surface[idx] = self.water_surface[idx].max(self.effective_bed(idx));

                // Advect suspended sediment with flow
                // (simplified: sediment concentration moves with water)
            }
        }
    }

    fn update_settling(&mut self, dt: f32) {
        let cell_area = self.cell_size * self.cell_size;

        for idx in 0..self.terrain_height.len() {
            let depth = self.water_depth(idx);
            if depth < 0.001 {
                continue;
            }

            let conc = self.suspended_conc[idx];
            if conc < 0.0001 {
                continue;
            }

            // Volume of sediment that settles this timestep
            let settling_height = self.params.settling_velocity * dt;
            let settled_fraction = (settling_height / depth).min(1.0);
            let settled_volume = conc * cell_area * depth * settled_fraction;

            // Remove from suspension
            self.suspended_conc[idx] *= 1.0 - settled_fraction;

            // Add to bed (accounting for porosity)
            let solid_fraction = 1.0 - self.params.bed_porosity;
            let bed_height_increase = settled_volume / (cell_area * solid_fraction);
            self.sediment_height[idx] += bed_height_increase;

            // Track total deposited
            self.total_sediment_deposited += settled_volume;

            // Cap sediment height at water surface (forms beach)
            let max_sed = self.water_surface[idx] - self.terrain_height[idx];
            if self.sediment_height[idx] > max_sed {
                self.sediment_height[idx] = max_sed.max(0.0);
            }
        }
    }

    fn update_diffusion(&mut self, dt: f32) {
        // Simple diffusion of suspended sediment (smooths concentration)
        let rate = self.params.diffusion_rate * dt;
        let mut new_conc = self.suspended_conc.clone();

        for z in 1..self.depth - 1 {
            for x in 1..self.width - 1 {
                let idx = self.idx(x, z);
                let c = self.suspended_conc[idx];

                let c_l = self.suspended_conc[self.idx(x - 1, z)];
                let c_r = self.suspended_conc[self.idx(x + 1, z)];
                let c_b = self.suspended_conc[self.idx(x, z - 1)];
                let c_f = self.suspended_conc[self.idx(x, z + 1)];

                let laplacian = c_l + c_r + c_b + c_f - 4.0 * c;
                new_conc[idx] = (c + rate * laplacian).max(0.0);
            }
        }

        self.suspended_conc = new_conc;
    }

    fn handle_overflow(&mut self) {
        // Check for cells where water exceeds overflow height
        for idx in 0..self.terrain_height.len() {
            if self.water_surface[idx] > self.params.overflow_height {
                // Remove excess water
                let excess = self.water_surface[idx] - self.params.overflow_height;
                self.water_surface[idx] = self.params.overflow_height;

                // Sediment in excess water is lost
                let lost_conc = self.suspended_conc[idx] * excess / self.water_depth(idx).max(0.001);
                self.overflow_sediment += lost_conc * self.cell_size * self.cell_size * excess;
            }
        }
    }

    /// Excavate settled tailings (spawns particles for re-processing)
    pub fn excavate(
        &mut self,
        world_x: f32,
        world_z: f32,
        radius: f32,
        dig_depth: f32
    ) -> Vec<(Vec3, f32)> {
        // Returns (position, sediment_volume) for particle spawning
        let mut spawns = Vec::new();

        let cx = ((world_x - self.world_offset.x) / self.cell_size) as i32;
        let cz = ((world_z - self.world_offset.z) / self.cell_size) as i32;
        let r_cells = (radius / self.cell_size).ceil() as i32;
        let r_sq = (radius / self.cell_size).powi(2);

        for dz in -r_cells..=r_cells {
            for dx in -r_cells..=r_cells {
                if (dx * dx + dz * dz) as f32 > r_sq {
                    continue;
                }

                let x = cx + dx;
                let z = cz + dz;

                if x < 0 || z < 0 || x >= self.width as i32 || z >= self.depth as i32 {
                    continue;
                }

                let idx = self.idx(x as usize, z as usize);
                let sed = self.sediment_height[idx];

                if sed > 0.0 {
                    let dig = dig_depth.min(sed);
                    self.sediment_height[idx] -= dig;

                    let volume = dig * self.cell_size * self.cell_size;
                    let world_pos = Vec3::new(
                        self.world_offset.x + (x as f32 + 0.5) * self.cell_size,
                        self.effective_bed(idx) + 0.1,
                        self.world_offset.z + (z as f32 + 0.5) * self.cell_size,
                    );

                    spawns.push((world_pos, volume));
                }
            }
        }

        spawns
    }
}
```

#### 2.2 Add to sim3d lib.rs

```rust
mod tailings;
pub use tailings::{TailingsPond, TailingsParams, TailingsStatus};
```

---

### Phase 3: Rendering

**Goal**: Visualize water surface and sediment bed.

#### 3.1 Water surface mesh

```rust
// Similar to heightfield rendering, but with:
// - Transparent blue material
// - Fresnel effect for realism
// - Depth-based color (deeper = darker)
```

#### 3.2 Sediment bed visualization

```rust
// Color sediment by:
// - Height (darker at base, lighter at top)
// - Age (fresh = light, compacted = darker brown)
// - Could show layers if tracking deposition history
```

#### 3.3 Suspended sediment (turbidity)

```rust
// Render water with turbidity based on suspended_conc
// High conc = murky brown
// Low conc = clear blue
```

---

### Phase 4: Active Zone Integration

**Goal**: Seamless particle ↔ heightfield transitions.

#### 4.1 Particle zone outflow → Tailings inflow

```rust
// At sluice outlet, where particles exit active zone:

fn transfer_to_tailings(
    particle_zone: &mut ParticleZone,
    tailings: &mut TailingsPond,
    outlet_position: Vec3,
    dt: f32,
) {
    // Count water and sediment particles exiting
    let mut water_volume = 0.0;
    let mut sediment_volume = 0.0;

    for particle in particle_zone.particles_near_outlet(outlet_position) {
        if particle.should_exit() {
            match particle.material {
                Material::Water => water_volume += particle.volume(),
                Material::Sand | Material::Silt => sediment_volume += particle.volume(),
                _ => {}
            }
            particle.mark_for_removal();
        }
    }

    // Add to tailings pond
    tailings.add_inflow(
        outlet_position.x,
        outlet_position.z,
        water_volume,
        sediment_volume,
    );
}
```

#### 4.2 Tailings excavation → Particle spawn

```rust
// When player digs tailings:

fn excavate_tailings_to_particles(
    tailings: &mut TailingsPond,
    particle_zone: &mut ParticleZone,
    dig_pos: Vec3,
    radius: f32,
    depth: f32,
) {
    let spawns = tailings.excavate(dig_pos.x, dig_pos.z, radius, depth);

    for (pos, volume) in spawns {
        // Convert volume to particles
        let particle_count = (volume / PARTICLE_VOLUME).ceil() as usize;

        for _ in 0..particle_count {
            particle_zone.spawn(Particle {
                position: pos + random_offset(),
                velocity: Vec3::ZERO,
                material: Material::Tailings,  // Special type, might contain gold!
                density: 1.8,  // Settled sediment density
            });
        }
    }
}
```

---

### Phase 5: Example Scene

**Goal**: Tailings pond demo with visible filling.

#### 5.1 Create tailings_demo.rs example

```rust
// crates/game/examples/tailings_demo.rs

// Features:
// 1. Sluice outlet feeding into tailings pond
// 2. Real-time sediment settling visualization
// 3. Status display (capacity remaining, fill %)
// 4. Ability to excavate tailings
// 5. Speed up time to watch pond fill
```

---

## File Summary

| File | Action |
|------|--------|
| `crates/sim3d/src/water_body.rs` | CREATE |
| `crates/sim3d/src/tailings.rs` | CREATE |
| `crates/sim3d/src/lib.rs` | MODIFY (add exports) |
| `crates/game/examples/tailings_demo.rs` | CREATE |
| `crates/game/src/rendering/water.rs` | CREATE (optional) |

---

## Integration with Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TERRAIN HEIGHTFIELD                      │
│              (dig_test.rs collapse mechanics)               │
└──────────────────────┬──────────────────────────────────────┘
                       │ dig() spawns particles
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    ACTIVE PARTICLE ZONE                     │
│              (FLIP/APIC + Drucker-Prager)                   │
│                                                             │
│   Excavator → particles → wash plant → sluice              │
└──────────────────────┬──────────────────────────────────────┘
                       │ outflow (water + sediment)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    TAILINGS POND                            │
│              (This plan - heightfield water + settling)     │
│                                                             │
│   • Water levels out                                        │
│   • Sediment settles → bed grows                           │
│   • Capacity decreases over time                           │
│   • Can excavate to re-process!                            │
└──────────────────────┬──────────────────────────────────────┘
                       │ excavate() spawns particles
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACK TO ACTIVE ZONE                      │
│              (Re-process old tailings for missed gold!)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Gameplay Loop

1. **Mine** → excavator digs terrain → particles
2. **Wash** → particles through wash plant → sluice
3. **Separate** → gold settles in sluice, waste exits
4. **Dispose** → waste water + sediment → tailings pond
5. **Fill up** → tailings pond gradually fills
6. **Manage** → build dam higher, or excavate, or new pond
7. **Re-process** → old tailings might have gold fines!

---

## Parameters to Tune

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `settling_velocity` | 0.01 m/s | 0.001-0.1 | How fast sediment settles |
| `bed_porosity` | 0.4 | 0.3-0.5 | How much bed compacts |
| `diffusion_rate` | 0.1 | 0.01-1.0 | Sediment concentration smoothing |
| `overflow_height` | 10.0 m | 5-50 | Dam crest height |

---

## Checklist

- [ ] Phase 1: Create `water_body.rs` with basic flow
- [ ] Phase 1: Test water leveling
- [ ] Phase 2: Create `tailings.rs` with settling
- [ ] Phase 2: Add status checking (Normal/Warning/Critical/Overflow)
- [ ] Phase 2: Add excavation function
- [ ] Phase 3: Water surface rendering
- [ ] Phase 3: Turbidity visualization
- [ ] Phase 4: Sluice outflow → tailings integration
- [ ] Phase 4: Tailings excavation → particle spawn
- [ ] Phase 5: Demo scene
\n\n---\n## ARCHIVED: 2026-01-09\n\n**Status:** Implemented\n\nLow-fidelity tailings system is now working.
