# Plan: Gold Panning Minigame

## Vision

A physics-based panning minigame that teaches players the core mechanics of density separation while being genuinely fun and skill-based. Players manipulate a small GPU FLIP simulation (64³-128³ grid) to separate gold from sand using realistic swirling, tilting, and overflow mechanics.

**Core loop:** Load dirt → Add water → Swirl to create vortex → Tilt to eject sand → Recover gold
**Skill curve:** Easy to learn (5 minutes), hard to master (better technique = higher recovery)
**Progression:** Tutorial → Field panning → Competition mode

---

## Why Start With Panning

1. **Smallest scope**: Single piece of equipment, contained simulation
2. **Proves core physics**: If density separation feels good here, it will everywhere
3. **Natural tutorial**: Players learn fluid dynamics through direct interaction
4. **Immediate feedback**: See results in seconds, not minutes
5. **Foundation for everything**: Sluice, jig, shaker all use same physics

**Implementation time estimate**: 2-3 days for playable prototype

---

## Physics Foundation (Already Have!)

### What We're Using

| System | Current Implementation | Used For |
|--------|----------------------|----------|
| **3D FLIP** | `gpu/flip_3d.rs` | Water swirling in pan |
| **Vorticity confinement** | `shaders/vorticity_confine_3d.wgsl` | Realistic eddy behavior |
| **Drift-flux sediment** | `gpu/g2p_3d.rs` | Density stratification |
| **Settling velocity** | Ferguson-Church model | Gold sinks, sand floats |
| **Particle size** | `Particle3D.density` | Different materials |

**Key insight:** We're just running a tiny FLIP sim (30cm diameter pan) instead of full sluice box!

---

## Pan Physics Model

### Geometry
```
Top View:              Side View:
    ____                 ___
  /      \              /   \___
 |  PAN   |            |    RIM \
  \______/              \_CREASE_/
   30cm Ø                8cm deep
```

**Specifications:**
- Diameter: 0.30m (standard 14" gold pan)
- Depth: 0.08m at center, 0.12m at rim
- Crease angle: 30° (where gold concentrates)
- Overflow rim: 0.08m height
- Material: Non-stick surface (low friction SDF)

### Forces Acting on Particles

#### 1. Gravity (Modified by Pan Tilt)
```rust
fn effective_gravity(tilt: Vec2) -> Vec3 {
    // Tilt forward/back and left/right
    Vec3::new(
        GRAVITY * tilt.x.sin(),    // X component from tilt
        -GRAVITY,                   // Downward
        GRAVITY * tilt.y.sin(),    // Z component from tilt
    )
}
```

**Player control:** Tilt angle -30° to +30° (forward dumps, back settles)

#### 2. Swirl (Centrifugal Vortex)
```rust
fn apply_swirl(grid: &mut FlipGrid3D, swirl_speed: f32, pan_center: Vec3) {
    for cell in grid.cells_mut() {
        let r = (cell.pos - pan_center).xz();  // Radial vector
        let dist = r.length();

        if dist < PAN_RADIUS {
            // Tangent velocity (perpendicular to radius)
            let tangent = Vec3::new(-r.y, 0.0, r.x).normalize();

            // Solid body rotation: v = ω × r
            let vortex_vel = tangent * swirl_speed * dist;

            // Inject into grid (vorticity confinement amplifies)
            cell.velocity += vortex_vel;
        }
    }
}
```

**Player control:** Swirl speed 0-120 RPM (30-60 RPM optimal for gold)

#### 3. Settling (Density-Dependent)
```rust
fn settling_velocity(particle: &Particle3D) -> f32 {
    // Ferguson-Church equation (already implemented!)
    let sg = particle.density;  // Specific gravity (gold=19.3, sand=2.65)
    let d = particle.diameter;  // Particle size (0.1mm to 10mm)

    // Shape factor (flaky gold settles slower)
    let shape_factor = if sg > 15.0 { 0.7 } else { 1.0 };

    shape_factor * ((sg - 1.0) * GRAVITY * d).sqrt()
}
```

**Physics result:** Gold settles at 0.15-0.45 m/s, sand at 0.02-0.08 m/s

#### 4. Turbulent Suspension
```rust
fn reduce_settling_in_vortex(particle: &mut Particle3D, vorticity_mag: f32) {
    // High vorticity keeps particles suspended
    let suspension_factor = (vorticity_mag / VORTICITY_THRESHOLD).min(1.0);

    // Reduce settling velocity in turbulent zones
    particle.settling_velocity *= 1.0 - suspension_factor * 0.8;
}
```

**Physics result:** Sand stays suspended in vortex, gold penetrates through

#### 5. Overflow (Material Ejection)
```rust
fn check_overflow(particle: &Particle3D, pan_center: Vec3) -> bool {
    let r = (particle.position - pan_center).xz().length();
    let rim_height = PAN_DEPTH - (r / PAN_RADIUS) * CREASE_DROP;

    // Particle washes over rim
    particle.position.y > rim_height
}
```

**Player control:** Tilt angle controls which material overflows first

---

## Material Types

### Sediment Classification
```rust
#[derive(Clone, Copy, Debug)]
pub enum PanMaterial {
    QuartzSand,   // SG 2.65, size 0.1-2mm, 60% of sample
    Magnetite,    // SG 5.2,  size 0.1-1mm, 20% (black sand indicator!)
    Garnet,       // SG 4.0,  size 0.2-1mm, 15%
    Gold,         // SG 19.3, size 0.1-5mm, 5% (the prize!)
}

impl PanMaterial {
    pub fn color(&self) -> [f32; 3] {
        match self {
            QuartzSand => [0.9, 0.85, 0.7],  // Tan/white
            Magnetite => [0.1, 0.1, 0.1],    // Black
            Garnet => [0.6, 0.2, 0.2],       // Red-brown
            Gold => [1.0, 0.85, 0.0],        // Yellow/orange
        }
    }

    pub fn specific_gravity(&self) -> f32 {
        match self {
            QuartzSand => 2.65,
            Magnetite => 5.2,
            Garnet => 4.0,
            Gold => 19.3,
        }
    }

    pub fn size_range(&self) -> (f32, f32) {
        match self {
            QuartzSand => (0.1, 2.0),   // mm
            Magnetite => (0.1, 1.0),
            Garnet => (0.2, 1.0),
            Gold => (0.1, 5.0),         // Fines to small nuggets
        }
    }
}
```

### Sample Composition
```rust
pub struct PanSample {
    total_mass: f32,           // grams (100g-500g typical)
    gold_content: f32,         // grams (0.01g-5g)
    gold_fineness: f32,        // 0-1000 (800-950 for placer gold)
    black_sand_ratio: f32,     // Magnetite indicator (high = good sign!)
    particle_count: usize,     // 1000-5000 particles
}

impl PanSample {
    pub fn from_terrain(terrain_cell: &TerrainCell) -> Self {
        let gold_ppm = terrain_cell.gold_ppm;
        let sample_mass = 250.0;  // grams

        Self {
            total_mass: sample_mass,
            gold_content: (gold_ppm / 1e6) * sample_mass,
            gold_fineness: 850.0 + rand(0..100),
            black_sand_ratio: 0.15 + rand(0.0..0.1),
            particle_count: 2000,
        }
    }

    pub fn spawn_particles(&self) -> Vec<Particle3D> {
        let mut particles = Vec::new();

        // 60% quartz sand
        for _ in 0..(self.particle_count as f32 * 0.6) as usize {
            particles.push(random_particle(PanMaterial::QuartzSand));
        }

        // 20% magnetite (black sand)
        for _ in 0..(self.particle_count as f32 * 0.2) as usize {
            particles.push(random_particle(PanMaterial::Magnetite));
        }

        // 15% garnet
        for _ in 0..(self.particle_count as f32 * 0.15) as usize {
            particles.push(random_particle(PanMaterial::Garnet));
        }

        // 5% gold (scaled by gold_content)
        let gold_particles = (self.gold_content / 0.01).min(100.0) as usize;
        for _ in 0..gold_particles {
            particles.push(random_particle(PanMaterial::Gold));
        }

        particles
    }
}
```

---

## Player Controls

### Input Scheme (Gamepad or Mouse)

#### Mouse/Touchpad
```rust
pub struct PanInput {
    // Tilt control
    tilt_x: f32,    // Mouse drag X → pan tilt left/right
    tilt_y: f32,    // Mouse drag Y → pan tilt forward/back

    // Swirl control
    swirl_speed: f32,  // Mouse wheel → swirl RPM (0-120)

    // Actions
    add_water: bool,   // Spacebar or click
    shake: bool,       // S key (quick agitation)
    dump: bool,        // D key (discard current contents)
}
```

#### Gamepad (Alternative)
```rust
// Left stick: Pan tilt
// Right stick: Camera
// RT: Swirl clockwise
// LT: Swirl counter-clockwise
// A: Add water
// B: Shake
// X: Dump
```

### Control Feel
```rust
fn update_pan_orientation(&mut self, input: &PanInput, dt: f32) {
    // Smooth tilt interpolation (not instant)
    let target_tilt = Vec2::new(input.tilt_x, input.tilt_y) * MAX_TILT;
    self.current_tilt = self.current_tilt.lerp(target_tilt, dt * TILT_SPEED);

    // Swirl acceleration (feels responsive but not instant)
    let target_swirl = input.swirl_speed;
    self.current_swirl += (target_swirl - self.current_swirl) * dt * SWIRL_ACCEL;
}
```

**Feel goals:**
- Tilt: Deliberate (0.2s to full tilt), like rotating a heavy pan
- Swirl: Responsive (0.1s to speed), like stirring water
- Water: Immediate splash when added

---

## Gameplay Loop

### Phase 1: Loading (5-10 seconds)
```
1. Player scoops dirt from stream/bucket
2. Particles spawn in pan center (random positions)
3. Camera zooms to pan view
4. Tutorial prompt: "Add water (Space)"
```

### Phase 2: Classifying (20-40 seconds)
```
1. Add water → pan fills to 60% depth
2. Swirl at medium speed (45 RPM)
   - Large rocks float to surface → shake to eject
   - Sand suspends in vortex
   - Gold sinks to bottom (visible flashes!)
3. Tilt forward slightly → coarse material overflows
4. Visual feedback: Water color changes (muddy → clearer)
```

### Phase 3: Concentrating (30-60 seconds)
```
1. Reduce swirl (30 RPM) → finer control
2. Tilt forward + swirl → eject suspended sand
3. Watch for black sand (magnetite) concentrating
   - "Getting close!" indicator when black sand visible
4. Gold appears in crease (flashing yellow particles)
5. Tilt back → gold settles in crease
```

### Phase 4: Recovery (10-20 seconds)
```
1. Final swirl at low speed (15 RPM)
2. Gentle forward tilt → last sand washes out
3. Stop swirl → gold and black sand in crease
4. Camera zoom to crease → count gold particles
5. Results screen: "Recovered 2.3g / 3.1g (74%)"
```

### Phase 5: Cleanup
```
1. Tap out gold into vial (collect animation)
2. Dump remaining black sand
3. Ready for next pan
```

**Total time:** 1-2 minutes per pan (realistic!)

---

## Skill Elements

### Beginner Mistakes
1. **Swirl too fast** → Gold washes over rim (20-40% loss)
2. **Tilt too far forward** → Everything dumps out (50-80% loss)
3. **Not enough water** → Material doesn't stratify (poor separation)
4. **Swirl too slow** → Sand doesn't suspend (slow progress)

### Expert Techniques
1. **Rhythm swirling** → Alternate fast/slow to eject sand layers
2. **Controlled tilt** → Small angle changes for precise ejection
3. **Water management** → Add water in pulses, not continuously
4. **Reading the pan** → Watch black sand concentration to know when gold is next

### Performance Metrics
```rust
pub struct PanningPerformance {
    recovery_rate: f32,      // % of gold retained (50-95%)
    time_taken: f32,         // seconds (faster = bonus)
    technique_score: f32,    // 0-100 based on movements
    gold_recovered: f32,     // grams
}

impl PanningPerformance {
    pub fn calculate_score(&self) -> u32 {
        let recovery_points = (self.recovery_rate * 100.0) as u32;
        let speed_bonus = if self.time_taken < 60.0 {
            ((60.0 - self.time_taken) * 2.0) as u32
        } else {
            0
        };
        let technique_bonus = self.technique_score as u32;

        recovery_points + speed_bonus + technique_bonus
    }
}
```

### Difficulty Progression
```rust
pub enum PanDifficulty {
    Tutorial,      // 10g gold in 250g sample (4%), coarse gold
    Easy,          // 2g gold in 250g sample (0.8%), mixed sizes
    Medium,        // 0.5g gold in 250g sample (0.2%), fine gold
    Hard,          // 0.1g gold in 500g sample (0.02%), flour gold + nugget
    Challenge,     // Time trial: 10 pans in 8 minutes
}
```

---

## Visual Feedback

### Particle Rendering
```rust
pub fn render_pan_particles(particles: &[Particle3D]) {
    for particle in particles {
        let color = particle.material.color();
        let size = particle.diameter * 10.0;  // Exaggerate for visibility

        // Special effects for gold
        if particle.material == PanMaterial::Gold {
            // Shimmer/glint effect
            let shimmer = (time * 10.0).sin() * 0.3 + 0.7;
            color *= shimmer;

            // Larger rendering size (easier to see)
            size *= 2.0;
        }

        // Black sand indicator (important!)
        if particle.material == PanMaterial::Magnetite {
            // Darker in water, visible when concentrated
            let visibility = particle.position.y / water_surface_height;
            color *= 0.3 + visibility * 0.7;
        }

        draw_sphere(particle.position, size, color);
    }
}
```

### Water Appearance
```rust
pub fn water_color_from_sediment(sediment_fraction: f32) -> [f32; 4] {
    // Clear → muddy brown based on suspended sediment
    let clarity = 1.0 - sediment_fraction;

    [
        0.2 + 0.4 * (1.0 - clarity),  // R: more brown when muddy
        0.4 + 0.3 * (1.0 - clarity),  // G:
        0.8 * clarity,                 // B: less blue when muddy
        0.6 + 0.3 * clarity,          // A: more transparent when clear
    ]
}
```

### UI Overlays
```
┌────────────────────────────────────────┐
│ GOLD PANNING - Classifier Creek       │
├────────────────────────────────────────┤
│                                        │
│       ╔═══════════════╗                │
│       ║   ~~~  ●  ~~~ ║ ← Vortex       │
│       ║  ○●●○○ ●● ○○  ║ ← Black sand   │
│       ║   ●●●●●●●●●   ║ ← Gold layer   │
│       ╚═══════════════╝                │
│                                        │
│  Tilt: ↑↓ ←→  [Mouse Drag]            │
│  Swirl: ◯ 45 RPM [Scroll Wheel]       │
│  Water: ▓▓▓▓▓░░░ 65%                  │
│                                        │
│  Gold visible: ●●● (3 flakes)         │
│  Black sand: Building up...           │
│                                        │
│  [Space] Add Water  [S] Shake  [D] Dump│
└────────────────────────────────────────┘
```

---

## Technical Implementation

### File Structure
```
crates/game/examples/
  panning_minigame.rs        # Main example binary

crates/game/src/
  panning/
    mod.rs                   # Public API
    pan_sim.rs               # Pan physics + FLIP integration
    pan_materials.rs         # Material types + properties
    pan_input.rs             # Player controls
    pan_renderer.rs          # Visual feedback
    pan_tutorial.rs          # Tutorial prompts
```

### Core Simulation Structure
```rust
pub struct PanSim {
    // Small GPU FLIP zone (64³ or 128³)
    flip: NarrowBandFlip3D,

    // Particle data (CPU, synced to GPU each frame)
    particles: Vec<Particle3D>,

    // Pan geometry
    pan_center: Vec3,
    pan_radius: f32,
    pan_depth: f32,

    // Player control state
    tilt_angle: Vec2,        // -30° to +30°
    swirl_speed: f32,        // 0-120 RPM
    water_level: f32,        // 0.0-1.0

    // Performance tracking
    gold_spawned: usize,
    gold_remaining: usize,
    time_elapsed: f32,

    // Visual state
    camera: Camera3D,
    ui_state: PanUI,
}

impl PanSim {
    pub fn new(sample: PanSample) -> Self {
        // Create small FLIP grid (just the pan area)
        let grid_size = 64;  // 64³ = 262k cells (tiny!)
        let cell_size = (PAN_DIAMETER * 1.5) / grid_size as f32;

        let flip = NarrowBandFlip3D::new(
            grid_size, grid_size, grid_size,
            cell_size,
            Vec3::ZERO,  // Centered on origin
        );

        // Spawn particles from sample
        let particles = sample.spawn_particles();

        Self {
            flip,
            particles,
            pan_center: Vec3::new(0.15, 0.04, 0.15),
            pan_radius: 0.15,
            pan_depth: 0.08,
            tilt_angle: Vec2::ZERO,
            swirl_speed: 0.0,
            water_level: 0.0,
            gold_spawned: particles.iter().filter(|p| p.is_gold()).count(),
            gold_remaining: 0,
            time_elapsed: 0.0,
            camera: Camera3D::default(),
            ui_state: PanUI::new(),
        }
    }

    pub fn update(&mut self, input: &PanInput, dt: f32) {
        self.time_elapsed += dt;

        // 1. Update player controls
        self.update_controls(input, dt);

        // 2. Apply forces to FLIP grid
        self.apply_gravity_tilt(dt);
        self.apply_swirl_vortex(dt);

        // 3. Run GPU FLIP step
        self.flip.step(dt);

        // 4. Sync particles with grid
        self.update_particles(dt);

        // 5. Check overflow
        self.remove_overflow_particles();

        // 6. Update UI
        self.update_ui();
    }

    fn apply_swirl_vortex(&mut self, dt: f32) {
        // Convert RPM to radians/second
        let omega = self.swirl_speed * 2.0 * PI / 60.0;

        // Inject vorticity into grid (uses existing vorticity confinement!)
        for cell in self.flip.grid.cells_mut() {
            let r = (cell.pos - self.pan_center).xz();
            let dist = r.length();

            if dist < self.pan_radius {
                let tangent = Vec3::new(-r.y, 0.0, r.x).normalize();
                let vortex_vel = tangent * omega * dist;

                // Add to cell velocity (vorticity confinement amplifies)
                cell.velocity += vortex_vel * dt;
            }
        }
    }

    fn update_particles(&mut self, dt: f32) {
        for particle in self.particles.iter_mut() {
            // Sample grid velocity (PIC-style for sediment)
            let v_grid = self.flip.grid.sample_velocity(particle.position);

            // Drag toward grid velocity
            let drag = 5.0;  // Higher for smaller particles
            particle.velocity = particle.velocity.lerp(v_grid, drag * dt);

            // Settling velocity (density-dependent)
            let settling = self.settling_velocity(particle);
            particle.velocity.y -= settling * dt;

            // Turbulent suspension (reduce settling in vortex)
            let vorticity_mag = self.flip.vorticity_magnitude(particle.position);
            if vorticity_mag > 2.0 {
                particle.velocity.y += settling * 0.7 * dt;  // Counteract settling
            }

            // Advect
            particle.position += particle.velocity * dt;

            // Collide with pan (simple SDF)
            self.collide_with_pan(particle);
        }
    }

    fn remove_overflow_particles(&mut self) {
        let pan_center = self.pan_center;
        let pan_radius = self.pan_radius;
        let pan_depth = self.pan_depth;

        self.particles.retain(|p| {
            // Distance from pan center
            let r = (p.position - pan_center).xz().length();

            // Rim height (lower at edges for overflow)
            let rim_height = pan_depth - (r / pan_radius) * 0.04;

            // Keep if below rim
            p.position.y < pan_center.y + rim_height
        });

        // Update gold count
        self.gold_remaining = self.particles.iter()
            .filter(|p| p.material == PanMaterial::Gold)
            .count();
    }
}
```

### GPU Integration (Reuse Existing!)
```rust
// We already have everything needed:
// - flip_3d.rs: 3D FLIP solver ✓
// - vorticity_confine_3d.wgsl: Vortex amplification ✓
// - g2p_3d.wgsl: Particle advection with settling ✓
// - sdf_collision_3d.wgsl: Pan collision ✓

// Just need to:
// 1. Create small grid (64³ instead of 256³)
// 2. Define pan SDF geometry
// 3. Inject vorticity from swirl input
```

### Pan SDF (For Collision)
```wgsl
fn sdf_pan(pos: vec3<f32>) -> f32 {
    let pan_center = vec3<f32>(0.15, 0.04, 0.15);
    let pan_radius = 0.15;
    let pan_depth = 0.08;

    // Distance from center axis
    let r = length(pos.xz - pan_center.xz);

    // Height from bottom
    let h = pos.y - pan_center.y;

    // Pan profile: parabolic bottom + vertical rim
    let profile_h = pan_depth * (1.0 - (r / pan_radius) * (r / pan_radius));

    // SDF: negative inside, positive outside
    if r > pan_radius {
        return length(vec2<f32>(r - pan_radius, h)) - 0.005;  // Outside
    } else if h < 0.0 {
        return -h;  // Below bottom
    } else if h > profile_h {
        return h - profile_h;  // Above surface
    } else {
        return -min(h, profile_h - h);  // Inside
    }
}
```

---

## Tutorial Sequence

### Level 1: Classifier Pan (Learn Basics)
```
Sample: 10g gold in 250g dirt (super rich!)
Pan: 8" classifier (small, easy)
Goal: Recover >80% gold

Tutorial steps:
1. "Click to add water" → water fills pan
2. "Scroll up to swirl" → vortex appears
3. "Drag forward to tilt" → sand overflows
4. "Watch for black sand" → magnetite concentrates
5. "Slow swirl when gold visible" → gold settles
6. "Tilt back to catch gold" → gold in crease
7. "SUCCESS! 8.7g recovered (87%)"

Unlock: Standard 14" pan
```

### Level 2: Standard Pan (Learn Technique)
```
Sample: 2g gold in 250g dirt (realistic)
Pan: 14" standard gold pan
Goal: Recover >70% gold

Challenges:
- Less gold (need to be careful!)
- More material (slower panning)
- Finer gold (easier to lose)

Unlock: Multiple samples, creek locations
```

### Level 3: Competition Mode (Mastery)
```
Challenge: Pan 10 samples in 8 minutes
Goal: Recover >60% average, total >15g

Leaderboard:
1. GoldenEye - 87% - 22.3g
2. PanMaster - 81% - 19.7g
3. [YOU] - 74% - 18.1g

Rewards: Unlock equipment upgrades, new locations
```

---

## Progression & Rewards

### Pan Upgrades
```rust
pub enum PanType {
    Classifier,    // 8" test pan (tutorial)
    Standard,      // 14" standard ($25)
    Professional,  // 16" with riffles ($75)
    SuperPanner,   // 17" with textured bottom ($150)
}

impl PanType {
    pub fn capacity(&self) -> f32 {
        match self {
            Classifier => 100.0,    // grams
            Standard => 250.0,
            Professional => 400.0,
            SuperPanner => 500.0,
        }
    }

    pub fn bonus_recovery(&self) -> f32 {
        match self {
            Classifier => 0.0,
            Standard => 0.05,       // +5% recovery
            Professional => 0.10,   // +10% (riffles trap gold)
            SuperPanner => 0.15,    // +15% (texture + riffles)
        }
    }
}
```

### Location Unlocks
```rust
pub struct PanningLocation {
    name: String,
    gold_grade: f32,        // ppm (0.1-100)
    particle_size: f32,     // mm (0.1-10)
    black_sand_ratio: f32,  // 0-1
    difficulty: u32,        // 1-5
}

// Example locations:
let tutorial_creek = PanningLocation {
    name: "Classifier Creek".to_string(),
    gold_grade: 40.0,       // Super rich!
    particle_size: 2.0,     // Coarse (easy)
    black_sand_ratio: 0.3,
    difficulty: 1,
};

let eureka_gulch = PanningLocation {
    name: "Eureka Gulch".to_string(),
    gold_grade: 2.0,        // Decent
    particle_size: 0.5,     // Fine (harder)
    black_sand_ratio: 0.2,
    difficulty: 3,
};

let flour_gold_flat = PanningLocation {
    name: "Flour Gold Flat".to_string(),
    gold_grade: 0.2,        // Low grade
    particle_size: 0.1,     // Flour gold (very hard)
    black_sand_ratio: 0.4,  // High black sand (good indicator)
    difficulty: 5,
};
```

---

## Performance Targets

### Grid Size vs FPS
```
 64³ grid = 262k cells → 200+ FPS (overkill)
 96³ grid = 884k cells → 120+ FPS (smooth)
128³ grid = 2.1M cells →  60 FPS (target)
```

**Recommended:** Start with 64³, scale up if needed

### Particle Count
```
Tutorial:    500 particles (fast, simple)
Standard:   2000 particles (balanced)
Challenge:  5000 particles (demanding)
```

### Memory Usage
```
FLIP grid (64³):  ~10 MB (velocity + pressure + markers)
Particles (2000): ~200 KB (position, velocity, material)
GPU buffers:      ~15 MB total

Total: ~25 MB (negligible!)
```

---

## Implementation Phases

### Phase 0: Proof of Concept (4-6 hours)
- [x] Small FLIP grid (64³)
- [x] Spawn 500 particles (all sand)
- [x] Apply swirl (manual vorticity injection)
- [x] Visual: Simple sphere rendering

**Deliverable:** Water swirls, particles move

### Phase 1: Basic Panning (1-2 days)
- [ ] Pan SDF collision
- [ ] Tilt controls (gravity rotation)
- [ ] Swirl controls (RPM input)
- [ ] Overflow detection
- [ ] Material types (sand, magnetite, gold)
- [ ] Settling velocity by density

**Deliverable:** Playable panning with density separation

### Phase 2: Visual Polish (1 day)
- [ ] Gold shimmer effect
- [ ] Water turbidity (muddy → clear)
- [ ] Black sand visibility
- [ ] Camera zoom to gold in crease
- [ ] UI overlays (RPM, tilt, water level)

**Deliverable:** Visually satisfying experience

### Phase 3: Gameplay Systems (1 day)
- [ ] Sample generation from terrain
- [ ] Recovery percentage calculation
- [ ] Technique scoring
- [ ] Tutorial prompts
- [ ] Results screen

**Deliverable:** Complete minigame loop

### Phase 4: Progression (1 day)
- [ ] Multiple pan types
- [ ] Location system
- [ ] Leaderboard/stats
- [ ] Unlock conditions

**Deliverable:** Replayable with progression

---

## Testing Criteria

### Physics Validation
- [ ] Gold sinks faster than sand (visible in 2-3 seconds)
- [ ] Vortex suspends sand, gold penetrates (observable)
- [ ] Tilt forward → sand overflows first
- [ ] Black sand concentrates before gold appears
- [ ] Water clarity improves as sand is ejected

### Feel & Responsiveness
- [ ] Tilt feels weighty (not instant)
- [ ] Swirl responds quickly (not sluggish)
- [ ] Water addition is immediate (satisfying splash)
- [ ] Overflow is smooth (not sudden disappearance)

### Skill Expression
- [ ] Expert players consistently get >85% recovery
- [ ] Beginners get 40-60% (room to improve)
- [ ] Technique matters more than time
- [ ] Different samples require different strategies

### Performance
- [ ] 60+ FPS with 2000 particles
- [ ] <100ms input latency
- [ ] No stuttering during swirl
- [ ] Smooth particle rendering

---

## Risks & Mitigations

### Risk 1: Particles Escape Pan
**Problem:** Collision detection fails, particles fall through

**Mitigation:**
- Use conservative SDF with margin (0.005m)
- Add "sticky" crease (velocity damping near bottom)
- Failsafe: Delete particles below Y=-0.1m

### Risk 2: Gold Too Hard to See
**Problem:** Small gold particles invisible in muddy water

**Mitigation:**
- Exaggerate gold particle size (2x visual radius)
- Add shimmer/glint effect (pulsing brightness)
- UI indicator: "Gold visible: ●●● (3 flakes)"
- Camera auto-zoom when gold concentrates

### Risk 3: Physics Doesn't Feel Right
**Problem:** Settling too slow, vortex too weak, etc.

**Mitigation:**
- Tunable parameters exposed in UI (debug mode)
- Reference videos of real gold panning
- Playtest with non-technical users
- Iterate on "feel" not accuracy

### Risk 4: Performance Issues
**Problem:** 60 FPS target not met

**Mitigation:**
- Start with 64³ grid (overkill for pan size)
- Reduce particle count (500-1000 still looks good)
- LOD for distant particles (billboard sprites)
- Profile GPU (likely pressure solve is bottleneck)

---

## Success Metrics

Minigame is successful if:

1. **5-minute rule**: New players understand mechanics in <5 minutes
2. **Skill curve**: Recovery improves 10-20% after 30 minutes practice
3. **Replayability**: Players want to retry samples for better scores
4. **Foundation**: Physics translates to sluice/jig/shaker (same separation feel)
5. **Performance**: Maintains 60 FPS on target hardware

---

## Next Steps After Panning

Once panning minigame works:

1. **Sluice Box**: Same physics, continuous feed instead of batch
2. **Shaker Screen**: Add vibration force, screen mesh filtering
3. **Jig**: Add pulsating water velocity, stratification layers
4. **Trommel**: Add rotation, tumbling physics

All use same core: FLIP water + drift-flux sediment + density separation

---

## Files to Create

```
crates/game/examples/panning_minigame.rs      # Main binary
crates/game/src/panning/mod.rs                # Module root
crates/game/src/panning/sim.rs                # Pan physics
crates/game/src/panning/materials.rs          # Material definitions
crates/game/src/panning/controls.rs           # Input handling
crates/game/src/panning/camera.rs             # Camera system
crates/game/src/panning/ui.rs                 # UI overlays
crates/game/src/panning/tutorial.rs           # Tutorial system
crates/game/src/panning/progression.rs        # Unlocks & rewards
plans/panning-minigame.md                     # This file
```

---

## References

- Real panning videos: Search "gold panning technique" on YouTube
- [How a Gold Pan Works](https://www.911metallurgist.com/blog/how-to-pan-for-gold/)
- Our existing FLIP solver: `crates/game/src/gpu/flip_3d.rs`
- Vorticity confinement: `crates/game/src/gpu/shaders/vorticity_confine_3d.wgsl`
- Drift-flux sediment: `plans/gpu3d-slurry.md`
