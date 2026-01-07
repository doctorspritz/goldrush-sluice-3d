# Gold Panning Minigame - Detailed Implementation Steps

## Overview

This document provides step-by-step instructions to implement the gold panning minigame. Each step includes specific files to create/modify, exact code changes, and verification criteria.

**Time estimate:** 2-3 days for complete implementation
**Prerequisites:** Existing GPU FLIP solver in `crates/game/src/gpu/`

---

## Phase 0: Project Setup & Module Scaffolding

**Goal:** Create module structure and example binary
**Time:** 30 minutes
**Status:** ⬜ Not Started

### Step 0.1: Create Module Directory Structure

```bash
mkdir -p crates/game/src/panning
touch crates/game/src/panning/mod.rs
touch crates/game/src/panning/sim.rs
touch crates/game/src/panning/materials.rs
touch crates/game/src/panning/controls.rs
touch crates/game/examples/panning_minigame.rs
```

**Verify:** Files exist in correct locations

### Step 0.2: Create Module Root (`crates/game/src/panning/mod.rs`)

```rust
//! Gold panning minigame module.

pub mod sim;
pub mod materials;
pub mod controls;

pub use sim::PanSim;
pub use materials::{PanMaterial, PanSample};
pub use controls::PanInput;
```

**Verify:** Module compiles without errors

### Step 0.3: Export Module in Game Crate (`crates/game/src/lib.rs`)

Add to bottom of file:

```rust
#[cfg(feature = "panning")]
pub mod panning;
```

Update `Cargo.toml` features:

```toml
[features]
default = []
panning = []
```

**Verify:** `cargo build --features panning` succeeds

### Step 0.4: Create Minimal Example Binary

**File:** `crates/game/examples/panning_minigame.rs`

```rust
//! Gold panning minigame example.

use macroquad::prelude::*;

#[macroquad::main("Gold Panning")]
async fn main() {
    loop {
        clear_background(BLACK);

        draw_text("Panning Minigame - Coming Soon", 20.0, 40.0, 30.0, WHITE);

        next_frame().await
    }
}
```

**Verify:**
```bash
cargo run --example panning_minigame --features panning
```
Window opens with text displayed.

---

## Phase 1: Minimal Working Simulation

**Goal:** Spawn particles in pan, run FLIP sim, render basic view
**Time:** 4-6 hours
**Status:** ⬜ Not Started

### Step 1.1: Define Material Types

**File:** `crates/game/src/panning/materials.rs`

```rust
use glam::Vec3;
use rand::Rng;

/// Material types in pan
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanMaterial {
    QuartzSand,   // Light tan sand
    Magnetite,    // Black sand (indicator mineral)
    Gold,         // Yellow gold flakes
}

impl PanMaterial {
    /// Specific gravity relative to water
    pub fn specific_gravity(&self) -> f32 {
        match self {
            PanMaterial::QuartzSand => 2.65,
            PanMaterial::Magnetite => 5.2,
            PanMaterial::Gold => 19.3,
        }
    }

    /// Particle diameter range in meters
    pub fn size_range(&self) -> (f32, f32) {
        match self {
            PanMaterial::QuartzSand => (0.0001, 0.002),  // 0.1-2mm
            PanMaterial::Magnetite => (0.0001, 0.001),   // 0.1-1mm
            PanMaterial::Gold => (0.0002, 0.005),        // 0.2-5mm
        }
    }

    /// Visual color for rendering
    pub fn color(&self) -> [f32; 3] {
        match self {
            PanMaterial::QuartzSand => [0.9, 0.85, 0.7],  // Tan
            PanMaterial::Magnetite => [0.1, 0.1, 0.1],    // Black
            PanMaterial::Gold => [1.0, 0.85, 0.0],        // Gold
        }
    }
}

/// A single particle in the pan
#[derive(Clone, Copy, Debug)]
pub struct PanParticle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub material: PanMaterial,
    pub diameter: f32,
}

impl PanParticle {
    /// Create random particle of given material
    pub fn random(material: PanMaterial) -> Self {
        let mut rng = rand::thread_rng();
        let (min_d, max_d) = material.size_range();

        Self {
            position: Vec3::ZERO,  // Will be set by spawn location
            velocity: Vec3::ZERO,
            material,
            diameter: rng.gen_range(min_d..max_d),
        }
    }

    /// Specific gravity of this particle
    pub fn specific_gravity(&self) -> f32 {
        self.material.specific_gravity()
    }

    /// Visual color
    pub fn color(&self) -> [f32; 3] {
        self.material.color()
    }
}

/// Sample to pan (collection of particles)
#[derive(Clone, Debug)]
pub struct PanSample {
    pub total_mass_grams: f32,
    pub gold_content_grams: f32,
    pub particle_count: usize,
}

impl PanSample {
    /// Tutorial sample (rich, easy)
    pub fn tutorial() -> Self {
        Self {
            total_mass_grams: 250.0,
            gold_content_grams: 10.0,  // 4% - super rich!
            particle_count: 1000,
        }
    }

    /// Standard sample (realistic)
    pub fn standard() -> Self {
        Self {
            total_mass_grams: 250.0,
            gold_content_grams: 2.0,   // 0.8% - decent
            particle_count: 2000,
        }
    }

    /// Generate particles for this sample
    pub fn spawn_particles(&self) -> Vec<PanParticle> {
        let mut particles = Vec::with_capacity(self.particle_count);
        let mut rng = rand::thread_rng();

        // Spawn in pan center area
        let pan_center = Vec3::new(0.15, 0.02, 0.15);
        let spawn_radius = 0.08;

        // 60% quartz sand
        for _ in 0..(self.particle_count as f32 * 0.6) as usize {
            let mut p = PanParticle::random(PanMaterial::QuartzSand);
            p.position = pan_center + Self::random_in_disk(spawn_radius);
            particles.push(p);
        }

        // 25% magnetite
        for _ in 0..(self.particle_count as f32 * 0.25) as usize {
            let mut p = PanParticle::random(PanMaterial::Magnetite);
            p.position = pan_center + Self::random_in_disk(spawn_radius);
            particles.push(p);
        }

        // 15% gold (scaled by gold_content)
        let gold_count = ((self.gold_content_grams / 0.05) as usize).min(150);
        for _ in 0..gold_count {
            let mut p = PanParticle::random(PanMaterial::Gold);
            p.position = pan_center + Self::random_in_disk(spawn_radius);
            particles.push(p);
        }

        particles
    }

    fn random_in_disk(radius: f32) -> Vec3 {
        let mut rng = rand::thread_rng();
        let angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let r = rng.gen_range(0.0..radius) * rng.gen::<f32>().sqrt();

        Vec3::new(
            r * angle.cos(),
            rng.gen_range(-0.01..0.01),  // Small vertical variation
            r * angle.sin(),
        )
    }
}
```

**Verify:** `cargo build --features panning` succeeds

### Step 1.2: Define Pan Controls

**File:** `crates/game/src/panning/controls.rs`

```rust
use glam::Vec2;

/// Player input state for panning
#[derive(Clone, Copy, Debug, Default)]
pub struct PanInput {
    /// Pan tilt angle in radians (-0.5 to 0.5)
    /// X: left/right tilt
    /// Y: forward/back tilt
    pub tilt: Vec2,

    /// Swirl speed in RPM (0-120)
    pub swirl_rpm: f32,

    /// Add water (press spacebar)
    pub add_water: bool,

    /// Shake pan (press S)
    pub shake: bool,

    /// Dump contents (press D)
    pub dump: bool,
}

impl PanInput {
    /// Maximum tilt angle (radians)
    pub const MAX_TILT: f32 = 0.52;  // ~30 degrees

    /// Update from macroquad input
    pub fn update(&mut self) {
        use macroquad::prelude::*;

        // Mouse drag for tilt
        if is_mouse_button_down(MouseButton::Left) {
            let delta = mouse_delta_position();
            self.tilt.x += delta.x * 0.01;
            self.tilt.y -= delta.y * 0.01;  // Invert Y
        }

        // Clamp tilt
        self.tilt.x = self.tilt.x.clamp(-Self::MAX_TILT, Self::MAX_TILT);
        self.tilt.y = self.tilt.y.clamp(-Self::MAX_TILT, Self::MAX_TILT);

        // Mouse wheel for swirl
        let (_x, wheel_y) = mouse_wheel();
        self.swirl_rpm += wheel_y * 5.0;
        self.swirl_rpm = self.swirl_rpm.clamp(0.0, 120.0);

        // Keyboard actions
        self.add_water = is_key_pressed(KeyCode::Space);
        self.shake = is_key_pressed(KeyCode::S);
        self.dump = is_key_pressed(KeyCode::D);

        // Reset controls
        if is_key_pressed(KeyCode::R) {
            *self = Self::default();
        }
    }
}
```

**Verify:** `cargo build --features panning` succeeds

### Step 1.3: Create Pan Simulation Structure

**File:** `crates/game/src/panning/sim.rs`

```rust
use super::materials::{PanParticle, PanSample};
use super::controls::PanInput;
use glam::{Vec2, Vec3};

/// Pan simulation state
pub struct PanSim {
    /// Particles in pan
    pub particles: Vec<PanParticle>,

    /// Pan geometry
    pub pan_center: Vec3,
    pub pan_radius: f32,
    pub pan_depth: f32,

    /// Control state (smoothed)
    pub current_tilt: Vec2,
    pub current_swirl: f32,
    pub water_level: f32,

    /// Stats
    pub gold_spawned: usize,
    pub time_elapsed: f32,
}

impl PanSim {
    /// Create new pan simulation with sample
    pub fn new(sample: PanSample) -> Self {
        let particles = sample.spawn_particles();
        let gold_spawned = particles.iter()
            .filter(|p| p.material == super::materials::PanMaterial::Gold)
            .count();

        Self {
            particles,
            pan_center: Vec3::new(0.15, 0.04, 0.15),
            pan_radius: 0.15,  // 30cm diameter
            pan_depth: 0.08,   // 8cm deep
            current_tilt: Vec2::ZERO,
            current_swirl: 0.0,
            water_level: 0.0,
            gold_spawned,
            time_elapsed: 0.0,
        }
    }

    /// Update simulation
    pub fn update(&mut self, input: &PanInput, dt: f32) {
        self.time_elapsed += dt;

        // Smooth control interpolation
        self.update_controls(input, dt);

        // Simple particle physics (will be replaced with FLIP)
        self.update_particles_simple(dt);

        // Check overflow
        self.remove_overflow_particles();
    }

    fn update_controls(&mut self, input: &PanInput, dt: f32) {
        // Smooth tilt (0.2s to full tilt)
        let tilt_speed = 5.0;
        self.current_tilt = self.current_tilt.lerp(input.tilt, dt * tilt_speed);

        // Smooth swirl (0.1s to target RPM)
        let swirl_speed = 10.0;
        self.current_swirl += (input.swirl_rpm - self.current_swirl) * dt * swirl_speed;

        // Water management
        if input.add_water {
            self.water_level = (self.water_level + 0.3).min(1.0);
        }
    }

    fn update_particles_simple(&mut self, dt: f32) {
        // Simple gravity (will be replaced with FLIP)
        let gravity = self.effective_gravity();

        for particle in self.particles.iter_mut() {
            // Apply gravity
            particle.velocity += gravity * dt;

            // Drag (simulate water resistance)
            particle.velocity *= 0.95;

            // Update position
            particle.position += particle.velocity * dt;

            // Simple floor collision
            if particle.position.y < self.pan_center.y {
                particle.position.y = self.pan_center.y;
                particle.velocity.y = 0.0;
            }
        }
    }

    fn effective_gravity(&self) -> Vec3 {
        let base_gravity = 9.81;
        Vec3::new(
            base_gravity * self.current_tilt.x.sin(),
            -base_gravity,
            base_gravity * self.current_tilt.y.sin(),
        )
    }

    fn remove_overflow_particles(&mut self) {
        let pan_center = self.pan_center;
        let pan_radius = self.pan_radius;
        let pan_depth = self.pan_depth;

        self.particles.retain(|p| {
            let r = (p.position - pan_center).xz().length();
            let rim_height = pan_depth - (r / pan_radius) * 0.04;

            // Keep if below rim
            p.position.y < pan_center.y + rim_height
        });
    }

    /// Count remaining gold particles
    pub fn gold_remaining(&self) -> usize {
        self.particles.iter()
            .filter(|p| p.material == super::materials::PanMaterial::Gold)
            .count()
    }

    /// Recovery percentage
    pub fn recovery_percent(&self) -> f32 {
        if self.gold_spawned == 0 {
            return 0.0;
        }
        (self.gold_remaining() as f32 / self.gold_spawned as f32) * 100.0
    }
}
```

**Verify:** `cargo build --features panning` succeeds

### Step 1.4: Implement Example with Basic Rendering

**File:** `crates/game/examples/panning_minigame.rs`

```rust
//! Gold panning minigame example.

use macroquad::prelude::*;
use game::panning::{PanSim, PanSample, PanInput};

#[macroquad::main("Gold Panning")]
async fn main() {
    // Create simulation
    let mut sim = PanSim::new(PanSample::tutorial());
    let mut input = PanInput::default();

    // Camera setup
    let mut camera_angle = 0.0_f32;
    let camera_distance = 0.8;

    loop {
        let dt = get_frame_time().min(0.033);

        // Update input
        input.update();

        // Update simulation
        sim.update(&input, dt);

        // Camera rotation
        if is_key_down(KeyCode::Left) {
            camera_angle += dt;
        }
        if is_key_down(KeyCode::Right) {
            camera_angle -= dt;
        }

        // 3D view setup
        clear_background(Color::from_rgba(135, 206, 235, 255));

        let camera_pos = Vec3::new(
            sim.pan_center.x + camera_angle.cos() * camera_distance,
            sim.pan_center.y + 0.4,
            sim.pan_center.z + camera_angle.sin() * camera_distance,
        );

        set_camera(&Camera3D {
            position: camera_pos,
            target: sim.pan_center,
            up: Vec3::Y,
            fovy: 45.0,
            ..Default::default()
        });

        // Draw pan (simple cylinder)
        draw_cylinder(
            sim.pan_center,
            sim.pan_radius,
            sim.pan_depth * 0.1,
            None,
            Color::from_rgba(100, 100, 100, 255),
        );

        // Draw particles
        for particle in &sim.particles {
            let color = particle.color();
            let size = (particle.diameter * 1000.0).max(0.003);  // Min 3mm visual

            draw_sphere(
                particle.position,
                size,
                None,
                Color::from_rgba(
                    (color[0] * 255.0) as u8,
                    (color[1] * 255.0) as u8,
                    (color[2] * 255.0) as u8,
                    255,
                ),
            );
        }

        // 2D UI overlay
        set_default_camera();

        draw_text(
            &format!("Gold Panning - Tutorial"),
            20.0, 30.0, 24.0, WHITE,
        );
        draw_text(
            &format!("Tilt: {:.1}°, {:.1}°",
                input.tilt.x.to_degrees(),
                input.tilt.y.to_degrees()
            ),
            20.0, 60.0, 20.0, WHITE,
        );
        draw_text(
            &format!("Swirl: {:.0} RPM", input.swirl_rpm),
            20.0, 85.0, 20.0, WHITE,
        );
        draw_text(
            &format!("Gold: {}/{} ({:.1}%)",
                sim.gold_remaining(),
                sim.gold_spawned,
                sim.recovery_percent(),
            ),
            20.0, 110.0, 20.0, YELLOW,
        );
        draw_text(
            &format!("Particles: {}", sim.particles.len()),
            20.0, 135.0, 20.0, WHITE,
        );

        draw_text(
            "Controls:",
            20.0, 180.0, 18.0, LIGHTGRAY,
        );
        draw_text(
            "  Drag Mouse: Tilt Pan",
            20.0, 205.0, 16.0, LIGHTGRAY,
        );
        draw_text(
            "  Scroll Wheel: Swirl Speed",
            20.0, 225.0, 16.0, LIGHTGRAY,
        );
        draw_text(
            "  Space: Add Water",
            20.0, 245.0, 16.0, LIGHTGRAY,
        );
        draw_text(
            "  Arrow Keys: Rotate Camera",
            20.0, 265.0, 16.0, LIGHTGRAY,
        );
        draw_text(
            "  R: Reset",
            20.0, 285.0, 16.0, LIGHTGRAY,
        );

        next_frame().await
    }
}
```

**Verify:**
```bash
cargo run --example panning_minigame --features panning --release
```

Should see:
- ✓ Particles spawn in pan
- ✓ Particles fall to pan bottom
- ✓ Mouse drag tilts pan (particles slide)
- ✓ Scroll wheel changes swirl speed (number changes)
- ✓ Gold count displayed in UI
- ✓ Camera rotates with arrow keys

---

## Phase 2: GPU FLIP Integration

**Goal:** Replace simple physics with GPU FLIP simulation
**Time:** 4-6 hours
**Status:** ⬜ Not Started

### Step 2.1: Add GPU Dependencies to Pan Module

**File:** `crates/game/src/panning/sim.rs`

Add imports at top:

```rust
use crate::gpu::flip_3d::GpuFlip3D;
use crate::gpu::FlipParams3D;
use sim3d::Particle3D;  // Reuse existing particle type
```

### Step 2.2: Initialize Small FLIP Grid

Modify `PanSim::new()`:

```rust
pub struct PanSim {
    // ... existing fields ...

    /// GPU FLIP simulation (small grid for pan)
    flip: GpuFlip3D,

    /// FLIP parameters
    flip_params: FlipParams3D,
}

impl PanSim {
    pub fn new(sample: PanSample, device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // Small grid (64x64x64 for pan)
        let grid_size = 64;
        let cell_size = (PAN_DIAMETER * 1.5) / grid_size as f32;  // 0.007m cells

        let flip_params = FlipParams3D {
            width: grid_size,
            height: grid_size,
            depth: grid_size,
            cell_size,
            particle_radius: cell_size * 0.4,
            flip_ratio: 0.95,
            gravity: 9.81,
            dt: 0.016,
            iterations: 20,
            density_rest: 1000.0,
            vorticity_epsilon: 0.3,  // Strong vorticity for swirl
        };

        let flip = GpuFlip3D::new(device, flip_params.clone());

        // Convert PanParticles to Particle3D for FLIP
        let particles = sample.spawn_particles();
        let flip_particles: Vec<Particle3D> = particles.iter().map(|p| {
            Particle3D {
                position: p.position,
                velocity: p.velocity,
                affine_velocity: glam::Mat3::ZERO,
                old_grid_velocity: Vec3::ZERO,
                density: p.specific_gravity(),
            }
        }).collect();

        // ... rest of initialization
    }
}
```

### Step 2.3: Update Simulation Loop with FLIP

Replace `update_particles_simple()` with:

```rust
fn update_particles_flip(&mut self, dt: f32, device: &wgpu::Device, queue: &wgpu::Queue) {
    // Apply custom forces before FLIP step
    self.apply_tilt_gravity();
    self.apply_swirl_vorticity();

    // Run FLIP simulation step
    self.flip.step(device, queue, dt);

    // Sync particles back from GPU
    let gpu_particles = self.flip.get_particles(device, queue);

    // Update our particle tracking
    for (i, gpu_p) in gpu_particles.iter().enumerate() {
        if i < self.particles.len() {
            self.particles[i].position = gpu_p.position;
            self.particles[i].velocity = gpu_p.velocity;
        }
    }
}

fn apply_tilt_gravity(&mut self) {
    // Modify gravity vector based on tilt
    let tilted_gravity = Vec3::new(
        9.81 * self.current_tilt.x.sin(),
        -9.81,
        9.81 * self.current_tilt.y.sin(),
    );

    self.flip_params.gravity_vector = tilted_gravity;
}

fn apply_swirl_vorticity(&mut self) {
    // Inject vorticity into grid cells
    // Convert RPM to angular velocity
    let omega = self.current_swirl * 2.0 * std::f32::consts::PI / 60.0;

    // This will be done in GPU shader (see next step)
    self.flip_params.swirl_strength = omega;
}
```

### Step 2.4: Add Swirl Vorticity Shader

**File:** `crates/game/src/gpu/shaders/pan_swirl_3d.wgsl` (new)

```wgsl
// Apply swirl vorticity to pan simulation

struct SwirlParams {
    width: u32,
    height: u32,
    depth: u32,
    swirl_omega: f32,      // Angular velocity (rad/s)
    pan_center_x: f32,
    pan_center_y: f32,
    pan_center_z: f32,
    pan_radius: f32,
}

@group(0) @binding(0) var<uniform> params: SwirlParams;
@group(0) @binding(1) var<storage, read_write> velocity_u: array<f32>;
@group(0) @binding(2) var<storage, read_write> velocity_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocity_w: array<f32>;

fn idx_u(x: u32, y: u32, z: u32) -> u32 {
    return z * params.height * (params.width + 1) + y * (params.width + 1) + x;
}

fn idx_v(x: u32, y: u32, z: u32) -> u32 {
    return z * (params.height + 1) * params.width + y * params.width + x;
}

fn idx_w(x: u32, y: u32, z: u32) -> u32 {
    return z * params.height * params.width + y * params.width + x;
}

@compute @workgroup_size(8, 8, 8)
fn apply_swirl(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let x = global_id.x;
    let y = global_id.y;
    let z = global_id.z;

    if x >= params.width || y >= params.height || z >= params.depth {
        return;
    }

    // Cell center position
    let cell_x = f32(x) + 0.5;
    let cell_y = f32(y) + 0.5;
    let cell_z = f32(z) + 0.5;

    // Distance from pan center (XZ plane)
    let dx = cell_x - params.pan_center_x;
    let dz = cell_z - params.pan_center_z;
    let r = sqrt(dx * dx + dz * dz);

    // Only apply swirl inside pan radius
    if r < params.pan_radius {
        // Tangent direction (perpendicular to radius)
        let tangent_x = -dz;
        let tangent_z = dx;
        let tangent_mag = sqrt(tangent_x * tangent_x + tangent_z * tangent_z);

        if tangent_mag > 0.001 {
            // Solid body rotation: v = ω × r
            let vortex_strength = params.swirl_omega * r;
            let vortex_x = vortex_strength * tangent_x / tangent_mag;
            let vortex_z = vortex_strength * tangent_z / tangent_mag;

            // Add to velocity field
            if x < params.width {
                let idx = idx_u(x, y, z);
                velocity_u[idx] = velocity_u[idx] + vortex_x * 0.1;
            }
            if z < params.depth {
                let idx = idx_w(x, y, z);
                velocity_w[idx] = velocity_w[idx] + vortex_z * 0.1;
            }
        }
    }
}
```

**Verify:** Shader compiles without errors

### Step 2.5: Integrate Swirl Shader into FLIP

**File:** `crates/game/src/gpu/flip_3d.rs`

Add method:

```rust
pub fn apply_pan_swirl(
    &mut self,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    swirl_omega: f32,
    pan_center: Vec3,
    pan_radius: f32,
) {
    // Create swirl params buffer
    let params = SwirlParams {
        width: self.params.width,
        height: self.params.height,
        depth: self.params.depth,
        swirl_omega,
        pan_center_x: pan_center.x,
        pan_center_y: pan_center.y,
        pan_center_z: pan_center.z,
        pan_radius,
    };

    // Upload params and dispatch compute
    // ... (similar to other GPU passes)
}
```

Call in `PanSim::update()` before FLIP step:

```rust
// Apply swirl before FLIP step
self.flip.apply_pan_swirl(
    device,
    encoder,
    omega,
    self.pan_center,
    self.pan_radius,
);
```

**Verify:**
- Run panning example
- Increase swirl RPM with scroll wheel
- Should see particles begin to rotate/swirl

---

## Phase 3: Density-Based Settling

**Goal:** Heavy particles (gold) sink through vortex, light particles (sand) suspend
**Time:** 3-4 hours
**Status:** ⬜ Not Started

### Step 3.1: Add Settling Velocity Calculation

**File:** `crates/game/src/panning/materials.rs`

Add method to `PanParticle`:

```rust
impl PanParticle {
    /// Ferguson-Church settling velocity (m/s)
    pub fn settling_velocity(&self) -> f32 {
        let sg = self.specific_gravity();
        let d = self.diameter;
        let g = 9.81;

        // Shape factor (flaky gold settles slower)
        let shape_factor = if sg > 15.0 { 0.7 } else { 1.0 };

        // Ferguson-Church equation
        shape_factor * ((sg - 1.0) * g * d).sqrt()
    }

    /// Drag coefficient (affects how well particle follows water)
    pub fn drag_coefficient(&self) -> f32 {
        // Smaller, lighter particles follow water more closely
        let base_drag = 5.0;
        let size_factor = 1.0 / self.diameter.max(0.0001);
        let density_factor = 1.0 / self.specific_gravity();

        base_drag * size_factor * density_factor
    }
}
```

### Step 3.2: Modify G2P Shader for Sediment

**File:** `crates/game/src/gpu/shaders/g2p_3d.wgsl`

Modify particle update to handle density:

```wgsl
// In particle update loop:

let density = particles[i].density;

// Water particles (density ~ 1.0): Use FLIP
if density < 1.5 {
    // Existing FLIP/PIC blend
    let v_pic = sample_velocity(pos);
    let v_flip = particles[i].velocity + (v_pic - particles[i].old_grid_velocity);
    particles[i].velocity = mix(v_pic, v_flip, flip_ratio);
}
// Sediment particles (density > 1.5): Drift-flux
else {
    // PIC-style: Follow grid velocity
    let v_grid = sample_velocity(pos);

    // Drag toward grid velocity
    let drag = 5.0 / density;  // Heavier = less drag
    particles[i].velocity = mix(particles[i].velocity, v_grid, drag * dt);

    // Add settling velocity (downward)
    let settling = compute_settling_velocity(density, diameter);
    particles[i].velocity.y -= settling * dt;

    // Turbulent suspension (reduce settling in high vorticity)
    let vort_mag = sample_vorticity_magnitude(pos);
    if vort_mag > 2.0 {
        let suspension_factor = min(vort_mag / 10.0, 0.8);
        particles[i].velocity.y += settling * suspension_factor * dt;
    }
}
```

Add helper function:

```wgsl
fn compute_settling_velocity(density: f32, diameter: f32) -> f32 {
    let g = 9.81;
    let sg = density;

    // Shape factor (flaky gold)
    let shape_factor = select(1.0, 0.7, sg > 15.0);

    // Ferguson-Church
    return shape_factor * sqrt((sg - 1.0) * g * diameter);
}
```

### Step 3.3: Add Particle Diameter to GPU Buffers

**File:** `crates/game/src/gpu/flip_3d.rs`

Modify particle buffer layout:

```rust
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParticle {
    position: [f32; 4],        // xyz + padding
    velocity: [f32; 4],        // xyz + padding
    affine_c0: [f32; 4],       // APIC matrix row 0
    affine_c1: [f32; 4],       // APIC matrix row 1
    affine_c2: [f32; 4],       // APIC matrix row 2
    old_velocity: [f32; 4],    // For FLIP delta
    density: f32,              // NEW: Specific gravity
    diameter: f32,             // NEW: Particle size (m)
    _pad0: f32,
    _pad1: f32,
}
```

Update buffer upload in `PanSim`:

```rust
fn sync_particles_to_gpu(&mut self) {
    let gpu_particles: Vec<GpuParticle> = self.particles.iter().map(|p| {
        GpuParticle {
            position: [p.position.x, p.position.y, p.position.z, 0.0],
            velocity: [p.velocity.x, p.velocity.y, p.velocity.z, 0.0],
            // ... other fields ...
            density: p.specific_gravity(),
            diameter: p.diameter,
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }).collect();

    // Upload to GPU
    self.flip.upload_particles(device, queue, &gpu_particles);
}
```

**Verify:**
- Run panning example
- Swirl at medium speed (45 RPM)
- Gold particles should visibly sink to bottom
- Sand particles should stay suspended in vortex
- Black sand (magnetite) should be intermediate

---

## Phase 4: Visual Polish

**Goal:** Make physics visible and satisfying
**Time:** 2-3 hours
**Status:** ⬜ Not Started

### Step 4.1: Add Gold Shimmer Effect

**File:** `crates/game/examples/panning_minigame.rs`

Modify particle rendering:

```rust
// Draw particles with material-specific effects
for (i, particle) in sim.particles.iter().enumerate() {
    let mut color = particle.color();
    let mut size = (particle.diameter * 1000.0).max(0.003);

    // Gold shimmer
    if particle.material == PanMaterial::Gold {
        let shimmer = ((sim.time_elapsed * 3.0 + i as f32) % 2.0 - 1.0).abs();
        color[0] *= 0.7 + shimmer * 0.3;
        color[1] *= 0.7 + shimmer * 0.3;
        size *= 2.0;  // Make gold easier to see
    }

    // Black sand visibility
    if particle.material == PanMaterial::Magnetite {
        // Darker when submerged, visible when concentrated
        let depth_factor = (particle.position.y / sim.pan_depth).clamp(0.0, 1.0);
        color[0] *= 0.3 + depth_factor * 0.7;
        color[1] *= 0.3 + depth_factor * 0.7;
        color[2] *= 0.3 + depth_factor * 0.7;
    }

    draw_sphere(
        particle.position,
        size,
        None,
        Color::from_rgba(
            (color[0] * 255.0) as u8,
            (color[1] * 255.0) as u8,
            (color[2] * 255.0) as u8,
            255,
        ),
    );
}
```

### Step 4.2: Add Water Rendering

Add water surface rendering:

```rust
// Draw water surface (if water present)
if sim.water_level > 0.1 {
    // Calculate water clarity based on suspended particles
    let suspended_count = sim.particles.iter()
        .filter(|p| p.position.y > sim.pan_center.y + 0.02)
        .count();

    let turbidity = (suspended_count as f32 / sim.particles.len() as f32).min(1.0);
    let clarity = 1.0 - turbidity;

    // Water color: clear blue → muddy brown
    let water_color = Color::from_rgba(
        (50.0 + 150.0 * turbidity) as u8,   // More brown when muddy
        (100.0 + 100.0 * turbidity) as u8,
        (200.0 * clarity) as u8,            // Less blue when muddy
        (150.0 + 50.0 * clarity) as u8,     // More transparent when clear
    );

    let water_height = sim.pan_center.y + sim.pan_depth * sim.water_level;

    draw_cylinder(
        Vec3::new(sim.pan_center.x, water_height, sim.pan_center.z),
        sim.pan_radius * 0.95,
        0.001,  // Thin disk
        None,
        water_color,
    );
}
```

### Step 4.3: Enhance UI with Visual Indicators

```rust
// Enhanced UI
draw_rectangle(0.0, 0.0, 300.0, 350.0, Color::from_rgba(0, 0, 0, 180));

draw_text("GOLD PANNING", 20.0, 30.0, 28.0, GOLD);
draw_text("Tutorial Creek", 20.0, 55.0, 18.0, LIGHTGRAY);

// Control indicators with bars
draw_text("Tilt:", 20.0, 90.0, 20.0, WHITE);
draw_rectangle(80.0, 75.0, 150.0, 20.0, DARKGRAY);
let tilt_x_bar = (input.tilt.x / PanInput::MAX_TILT + 1.0) / 2.0 * 150.0;
draw_rectangle(80.0, 75.0, tilt_x_bar, 20.0, SKYBLUE);

draw_text("Swirl:", 20.0, 125.0, 20.0, WHITE);
draw_rectangle(80.0, 110.0, 150.0, 20.0, DARKGRAY);
let swirl_bar = (input.swirl_rpm / 120.0) * 150.0;
draw_rectangle(80.0, 110.0, swirl_bar, 20.0, SKYBLUE);

// Water level
draw_text("Water:", 20.0, 160.0, 20.0, WHITE);
draw_rectangle(80.0, 145.0, 150.0, 20.0, DARKGRAY);
let water_bar = sim.water_level * 150.0;
draw_rectangle(80.0, 145.0, water_bar, 20.0, BLUE);

// Gold indicator
draw_text(
    &format!("Gold: {}/{}", sim.gold_remaining(), sim.gold_spawned),
    20.0, 195.0, 22.0, GOLD,
);

let recovery = sim.recovery_percent();
let recovery_color = if recovery > 80.0 {
    GREEN
} else if recovery > 60.0 {
    YELLOW
} else {
    RED
};
draw_text(
    &format!("Recovery: {:.1}%", recovery),
    20.0, 220.0, 20.0, recovery_color,
);

// Black sand indicator (important!)
let magnetite_count = sim.particles.iter()
    .filter(|p| p.material == PanMaterial::Magnetite)
    .count();
if magnetite_count > 50 {
    draw_text("● Black sand concentrating!", 20.0, 250.0, 18.0, GRAY);
    if sim.gold_remaining() > 0 {
        draw_text("  → Gold nearby!", 20.0, 270.0, 16.0, GOLD);
    }
}

// Time
draw_text(
    &format!("Time: {:.0}s", sim.time_elapsed),
    20.0, 300.0, 18.0, LIGHTGRAY,
);
```

**Verify:**
- Gold particles shimmer and are easy to see
- Water changes color from clear → muddy based on suspended sediment
- Black sand indicator appears when magnetite concentrates
- UI bars show control values intuitively
- Recovery percentage color-codes performance

---

## Phase 5: Gameplay Systems

**Goal:** Tutorial, scoring, progression
**Time:** 3-4 hours
**Status:** ⬜ Not Started

### Step 5.1: Add Tutorial State Machine

**File:** `crates/game/src/panning/tutorial.rs` (new)

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TutorialStep {
    Welcome,
    AddWater,
    Swirl,
    Tilt,
    WatchBlackSand,
    SlowDown,
    Recovery,
    Complete,
}

pub struct Tutorial {
    current_step: TutorialStep,
    step_start_time: f32,
    total_time: f32,
}

impl Tutorial {
    pub fn new() -> Self {
        Self {
            current_step: TutorialStep::Welcome,
            step_start_time: 0.0,
            total_time: 0.0,
        }
    }

    pub fn update(&mut self, sim: &PanSim, input: &PanInput, dt: f32) {
        self.total_time += dt;
        let step_time = self.total_time - self.step_start_time;

        match self.current_step {
            TutorialStep::Welcome => {
                if step_time > 2.0 {
                    self.advance_step();
                }
            }
            TutorialStep::AddWater => {
                if sim.water_level > 0.5 {
                    self.advance_step();
                }
            }
            TutorialStep::Swirl => {
                if input.swirl_rpm > 30.0 && step_time > 3.0 {
                    self.advance_step();
                }
            }
            TutorialStep::Tilt => {
                if input.tilt.y > 0.1 && step_time > 3.0 {
                    self.advance_step();
                }
            }
            TutorialStep::WatchBlackSand => {
                let magnetite_visible = sim.particles.iter()
                    .filter(|p| p.material == super::materials::PanMaterial::Magnetite)
                    .filter(|p| p.position.y < sim.pan_center.y + 0.02)
                    .count() > 30;

                if magnetite_visible {
                    self.advance_step();
                }
            }
            TutorialStep::SlowDown => {
                if input.swirl_rpm < 20.0 && step_time > 5.0 {
                    self.advance_step();
                }
            }
            TutorialStep::Recovery => {
                if sim.recovery_percent() > 70.0 {
                    self.advance_step();
                }
            }
            TutorialStep::Complete => {
                // Stay here
            }
        }
    }

    fn advance_step(&mut self) {
        self.current_step = match self.current_step {
            TutorialStep::Welcome => TutorialStep::AddWater,
            TutorialStep::AddWater => TutorialStep::Swirl,
            TutorialStep::Swirl => TutorialStep::Tilt,
            TutorialStep::Tilt => TutorialStep::WatchBlackSand,
            TutorialStep::WatchBlackSand => TutorialStep::SlowDown,
            TutorialStep::SlowDown => TutorialStep::Recovery,
            TutorialStep::Recovery => TutorialStep::Complete,
            TutorialStep::Complete => TutorialStep::Complete,
        };

        self.step_start_time = self.total_time;
    }

    pub fn get_prompt(&self) -> &str {
        match self.current_step {
            TutorialStep::Welcome => "Welcome to gold panning! Let's learn the basics.",
            TutorialStep::AddWater => "Press SPACE to add water to the pan",
            TutorialStep::Swirl => "SCROLL UP to create a swirl (aim for 30-60 RPM)",
            TutorialStep::Tilt => "DRAG MOUSE FORWARD to tilt pan and eject sand",
            TutorialStep::WatchBlackSand => "Watch for black sand (magnetite) concentrating...",
            TutorialStep::SlowDown => "SCROLL DOWN to slow swirl - gold will settle",
            TutorialStep::Recovery => "Great! Try to get >70% recovery",
            TutorialStep::Complete => "Tutorial complete! Press R to try again",
        }
    }

    pub fn is_complete(&self) -> bool {
        self.current_step == TutorialStep::Complete
    }
}
```

### Step 5.2: Add Performance Scoring

**File:** `crates/game/src/panning/sim.rs`

Add to `PanSim`:

```rust
pub struct PanPerformance {
    pub recovery_rate: f32,      // %
    pub time_taken: f32,         // seconds
    pub technique_score: f32,    // 0-100
    pub gold_recovered_grams: f32,
}

impl PanSim {
    pub fn calculate_performance(&self) -> PanPerformance {
        let recovery_rate = self.recovery_percent();
        let time_taken = self.time_elapsed;

        // Technique scoring based on control smoothness
        let avg_tilt = (self.current_tilt.x.abs() + self.current_tilt.y.abs()) / 2.0;
        let optimal_tilt = 0.2;  // ~11 degrees
        let tilt_score = 100.0 * (1.0 - (avg_tilt - optimal_tilt).abs() / optimal_tilt).max(0.0);

        let optimal_swirl = 45.0;  // RPM
        let swirl_score = 100.0 * (1.0 - (self.current_swirl - optimal_swirl).abs() / optimal_swirl).max(0.0);

        let technique_score = (tilt_score + swirl_score) / 2.0;

        // Estimate gold recovered (assuming 0.05g per particle)
        let gold_recovered_grams = self.gold_remaining() as f32 * 0.05;

        PanPerformance {
            recovery_rate,
            time_taken,
            technique_score,
            gold_recovered_grams,
        }
    }

    pub fn calculate_score(&self) -> u32 {
        let perf = self.calculate_performance();

        let recovery_points = perf.recovery_rate as u32;

        let speed_bonus = if perf.time_taken < 60.0 {
            ((60.0 - perf.time_taken) * 2.0) as u32
        } else {
            0
        };

        let technique_bonus = perf.technique_score as u32;

        recovery_points + speed_bonus + technique_bonus
    }
}
```

### Step 5.3: Add Results Screen

**File:** `crates/game/examples/panning_minigame.rs`

Add game state:

```rust
enum GameState {
    Tutorial(Tutorial),
    Playing,
    Results(PanPerformance),
}

// In main loop:
let mut game_state = GameState::Tutorial(Tutorial::new());

// Update based on state
match &mut game_state {
    GameState::Tutorial(tutorial) => {
        tutorial.update(&sim, &input, dt);

        // Draw tutorial prompt
        draw_rectangle(
            screen_width() / 2.0 - 300.0,
            screen_height() - 100.0,
            600.0,
            80.0,
            Color::from_rgba(0, 0, 0, 200),
        );
        draw_text_ex(
            tutorial.get_prompt(),
            screen_width() / 2.0 - 280.0,
            screen_height() - 60.0,
            TextParams {
                font_size: 24,
                color: YELLOW,
                ..Default::default()
            },
        );

        if tutorial.is_complete() && is_key_pressed(KeyCode::R) {
            game_state = GameState::Playing;
            sim = PanSim::new(PanSample::standard(), &device, &queue);
        }
    }

    GameState::Playing => {
        // Check for completion (user decides when done)
        if is_key_pressed(KeyCode::Enter) {
            let performance = sim.calculate_performance();
            game_state = GameState::Results(performance);
        }
    }

    GameState::Results(perf) => {
        // Draw results screen
        draw_rectangle(
            screen_width() / 2.0 - 250.0,
            screen_height() / 2.0 - 200.0,
            500.0,
            400.0,
            Color::from_rgba(20, 20, 20, 240),
        );

        draw_text("RESULTS",
            screen_width() / 2.0 - 100.0,
            screen_height() / 2.0 - 160.0,
            40.0, GOLD,
        );

        draw_text(
            &format!("Gold Recovered: {:.2}g", perf.gold_recovered_grams),
            screen_width() / 2.0 - 200.0,
            screen_height() / 2.0 - 100.0,
            24.0, YELLOW,
        );

        draw_text(
            &format!("Recovery: {:.1}%", perf.recovery_rate),
            screen_width() / 2.0 - 200.0,
            screen_height() / 2.0 - 60.0,
            24.0, WHITE,
        );

        draw_text(
            &format!("Time: {:.0}s", perf.time_taken),
            screen_width() / 2.0 - 200.0,
            screen_height() / 2.0 - 20.0,
            24.0, WHITE,
        );

        draw_text(
            &format!("Technique: {:.0}/100", perf.technique_score),
            screen_width() / 2.0 - 200.0,
            screen_height() / 2.0 + 20.0,
            24.0, WHITE,
        );

        let total_score = sim.calculate_score();
        draw_text(
            &format!("TOTAL SCORE: {}", total_score),
            screen_width() / 2.0 - 200.0,
            screen_height() / 2.0 + 80.0,
            32.0, GOLD,
        );

        draw_text(
            "Press R to pan again",
            screen_width() / 2.0 - 150.0,
            screen_height() / 2.0 + 150.0,
            20.0, LIGHTGRAY,
        );

        if is_key_pressed(KeyCode::R) {
            game_state = GameState::Playing;
            sim = PanSim::new(PanSample::standard(), &device, &queue);
        }
    }
}
```

**Verify:** Complete gameplay loop with tutorial → playing → results → retry

---

## Testing Checklist

### Phase 0: Setup ✓
- [ ] Module structure created
- [ ] Example binary runs
- [ ] No compile errors

### Phase 1: Basic Sim ✓
- [ ] Particles spawn in pan center
- [ ] Particles fall to pan bottom
- [ ] Mouse drag tilts pan
- [ ] Particles slide when tilted
- [ ] Gold count displays correctly
- [ ] Camera rotates with arrow keys

### Phase 2: FLIP Integration ✓
- [ ] FLIP grid initializes (64³)
- [ ] Particles sync to GPU
- [ ] Water simulation runs smoothly
- [ ] Swirl creates visible rotation
- [ ] Vorticity amplifies over time
- [ ] 60+ FPS maintained

### Phase 3: Settling ✓
- [ ] Gold sinks faster than sand (observable in 2-3 seconds)
- [ ] Sand suspends in vortex
- [ ] Magnetite intermediate behavior
- [ ] Turbulence reduces settling
- [ ] Different particle sizes affect fall rate

### Phase 4: Visuals ✓
- [ ] Gold shimmers/glints
- [ ] Water color changes with turbidity
- [ ] Black sand becomes visible when concentrated
- [ ] UI bars update smoothly
- [ ] Recovery percentage color-coded

### Phase 5: Gameplay ✓
- [ ] Tutorial guides new players
- [ ] Steps advance automatically
- [ ] Results screen shows performance
- [ ] Scoring rewards good technique
- [ ] Can retry samples

---

## Performance Targets

| Metric | Target | How to Measure |
|--------|--------|----------------|
| FPS | 60+ | Macroquad frame time |
| Input latency | <100ms | Mouse drag responsiveness |
| Particle count | 2000 | Tutorial sample |
| Grid size | 64³ (262k cells) | Initial config |
| Swirl responsiveness | <0.5s to visible | Scroll wheel → rotation |
| Settling time (gold) | 2-3 seconds | Visual observation |

---

## Troubleshooting

### Problem: Particles fall through pan bottom
**Solution:**
- Check pan SDF collision in shader
- Reduce timestep if velocities too high
- Add sticky floor (velocity damping when y < pan_center.y + 0.01)

### Problem: Swirl too weak/not visible
**Solution:**
- Increase vorticity confinement epsilon (0.5-1.0)
- Increase swirl injection strength in shader
- Reduce water damping

### Problem: Gold doesn't settle
**Solution:**
- Check settling velocity calculation (should be ~0.3 m/s for gold)
- Reduce turbulent suspension factor
- Check particle density is correct (19.3 for gold)

### Problem: FPS drops below 60
**Solution:**
- Reduce grid size to 48³ or 32³
- Reduce particle count
- Reduce pressure solver iterations
- Profile GPU (likely pressure solve)

### Problem: Controls feel sluggish
**Solution:**
- Increase interpolation speed constants
- Reduce input smoothing
- Check frame time stability

---

## Next Steps After Completion

1. **Polish:**
   - Add sound effects (swirl, water splash, gold clink)
   - Particle trails for motion visibility
   - Camera shake on pan shake action

2. **Content:**
   - Multiple sample types (rich, poor, fine gold)
   - Different pan types (classifier, standard, professional)
   - Locations with different grades

3. **Progression:**
   - Unlock system
   - Leaderboards
   - Achievements (speed run, perfect recovery)

4. **Scale Up:**
   - Use panning physics for sluice box
   - Add trommel with rotation
   - Implement jig with pulsation

---

## Success Criteria

Minigame is complete when:

1. ✓ Tutorial teaches mechanics in <5 minutes
2. ✓ Gold visibly settles through vortex
3. ✓ Sand visibly suspends and ejects
4. ✓ Black sand indicator works
5. ✓ Recovery >80% achievable with skill
6. ✓ Results screen shows meaningful feedback
7. ✓ 60 FPS maintained with 2000 particles
8. ✓ Controls feel responsive and intuitive

---

## Time Estimates Summary

| Phase | Time | Cumulative |
|-------|------|------------|
| 0: Setup | 30 min | 30 min |
| 1: Basic Sim | 4-6 hours | 5-7 hours |
| 2: FLIP Integration | 4-6 hours | 9-13 hours |
| 3: Settling | 3-4 hours | 12-17 hours |
| 4: Visuals | 2-3 hours | 14-20 hours |
| 5: Gameplay | 3-4 hours | 17-24 hours |

**Total: 2-3 working days**

---

## Files Created/Modified Summary

### New Files
```
crates/game/src/panning/mod.rs
crates/game/src/panning/sim.rs
crates/game/src/panning/materials.rs
crates/game/src/panning/controls.rs
crates/game/src/panning/tutorial.rs
crates/game/examples/panning_minigame.rs
crates/game/src/gpu/shaders/pan_swirl_3d.wgsl
```

### Modified Files
```
crates/game/src/lib.rs
crates/game/Cargo.toml
crates/game/src/gpu/flip_3d.rs
crates/game/src/gpu/shaders/g2p_3d.wgsl
```

### Lines of Code (Estimate)
- Module code: ~1200 lines
- Shaders: ~200 lines
- Example: ~400 lines
- **Total: ~1800 lines**
