# Task 2: Transfer Zone Types

**File to create:** `crates/game/src/washplant/transfer.rs`

## Goal
Define the TransferZone system for moving particles between stages.

## Types to Implement

```rust
use glam::{Mat3, Vec3};

/// Axis-aligned bounding box for capture region
#[derive(Clone, Copy, Debug)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x && point.x <= self.max.x &&
        point.y >= self.min.y && point.y <= self.max.y &&
        point.z >= self.min.z && point.z <= self.max.z
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }
}

/// A particle in transit between stages
#[derive(Clone, Debug)]
pub struct TransitParticle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub density: f32,
    pub remaining_time: f32,
    /// True if this is a sediment particle (vs water)
    pub is_sediment: bool,
}

/// Defines how particles transfer from one stage to another
#[derive(Clone, Debug)]
pub struct TransferZone {
    /// Source stage index
    pub from_stage: usize,
    /// Destination stage index
    pub to_stage: usize,

    /// Region in source stage where particles are captured
    /// Coordinates are in source stage's local space
    pub capture_aabb: AABB,

    /// Direction particles must be moving to be captured
    /// Particles with velocity.dot(exit_direction) > 0 are captured
    pub exit_direction: Vec3,

    /// Position where particles enter destination stage
    /// In destination stage's local space
    pub inject_position: Vec3,

    /// Velocity applied to particles on injection
    pub inject_velocity: Vec3,

    /// Optional velocity transform (for angled transfers)
    pub velocity_transform: Option<Mat3>,

    /// Time delay before particle appears in destination (seconds)
    pub transit_time: f32,

    /// Particles currently in transit
    transit_queue: Vec<TransitParticle>,
}

impl TransferZone {
    pub fn new(
        from_stage: usize,
        to_stage: usize,
        capture_aabb: AABB,
        exit_direction: Vec3,
        inject_position: Vec3,
    ) -> Self {
        Self {
            from_stage,
            to_stage,
            capture_aabb,
            exit_direction: exit_direction.normalize(),
            inject_position,
            inject_velocity: Vec3::ZERO,
            velocity_transform: None,
            transit_time: 0.1, // 100ms default
            transit_queue: Vec::new(),
        }
    }

    pub fn with_inject_velocity(mut self, velocity: Vec3) -> Self {
        self.inject_velocity = velocity;
        self
    }

    pub fn with_transit_time(mut self, time: f32) -> Self {
        self.transit_time = time;
        self
    }

    pub fn with_velocity_transform(mut self, transform: Mat3) -> Self {
        self.velocity_transform = Some(transform);
        self
    }

    /// Check if a particle should be captured
    pub fn should_capture(&self, position: Vec3, velocity: Vec3) -> bool {
        self.capture_aabb.contains(position) && velocity.dot(self.exit_direction) > 0.0
    }

    /// Add a particle to the transit queue
    pub fn enqueue(&mut self, position: Vec3, velocity: Vec3, density: f32, is_sediment: bool) {
        let transformed_vel = match self.velocity_transform {
            Some(t) => t * velocity,
            None => velocity,
        };

        self.transit_queue.push(TransitParticle {
            position: self.inject_position,
            velocity: self.inject_velocity + transformed_vel * 0.5, // Blend inject + incoming
            density,
            remaining_time: self.transit_time,
            is_sediment,
        });
    }

    /// Update transit timers and return particles ready to inject
    pub fn tick(&mut self, dt: f32) -> Vec<TransitParticle> {
        let mut ready = Vec::new();

        self.transit_queue.retain_mut(|p| {
            p.remaining_time -= dt;
            if p.remaining_time <= 0.0 {
                ready.push(p.clone());
                false
            } else {
                true
            }
        });

        ready
    }

    /// Number of particles currently in transit
    pub fn in_transit(&self) -> usize {
        self.transit_queue.len()
    }

    /// Clear all particles in transit
    pub fn clear(&mut self) {
        self.transit_queue.clear();
    }
}

/// Statistics about transfers
#[derive(Clone, Debug, Default)]
pub struct TransferStats {
    pub particles_captured: u64,
    pub particles_injected: u64,
    pub water_transferred: u64,
    pub sediment_transferred: u64,
}
```

## Update mod.rs

Add to `crates/game/src/washplant/mod.rs`:
```rust
mod transfer;
pub use transfer::*;
```

## Testing
Run `cargo check -p game` to verify compilation.
