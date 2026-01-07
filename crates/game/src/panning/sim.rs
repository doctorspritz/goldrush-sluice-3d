use super::controls::PanInput;
use super::materials::{PanMaterial, PanParticle, PanSample};
use glam::{Vec2, Vec3};
use rand::Rng;

/// Pan simulation state.
pub struct PanSim {
    pub particles: Vec<PanParticle>,
    pub pan_center: Vec3,
    pub pan_radius: f32,
    pub pan_depth: f32,
    pub current_tilt: Vec2,
    pub current_swirl: f32,
    pub water_level: f32,
    pub gold_spawned: usize,
    pub time_elapsed: f32,
}

impl PanSim {
    pub fn new(sample: PanSample) -> Self {
        let particles = sample.spawn_particles();
        let gold_spawned = particles
            .iter()
            .filter(|p| p.material == PanMaterial::Gold)
            .count();

        Self {
            particles,
            pan_center: Vec3::new(0.30, 0.04, 0.30),
            pan_radius: 0.30,
            pan_depth: 0.16,
            current_tilt: Vec2::ZERO,
            current_swirl: 0.0,
            water_level: 0.0,
            gold_spawned,
            time_elapsed: 0.0,
        }
    }

    pub fn update(&mut self, input: &PanInput, dt: f32) {
        self.time_elapsed += dt;
        self.update_controls(input, dt);
        self.update_particles_simple(dt, input.shake);
        if input.dump {
            self.particles.clear();
        } else {
            self.remove_overflow_particles();
        }
    }

    pub fn update_controls_only(&mut self, input: &PanInput, dt: f32) {
        self.time_elapsed += dt;
        self.update_controls(input, dt);
    }

    pub fn cull_overflow(&mut self) {
        self.remove_overflow_particles();
    }

    fn update_controls(&mut self, input: &PanInput, dt: f32) {
        let tilt_speed = 5.0;
        self.current_tilt = self.current_tilt.lerp(input.tilt, dt * tilt_speed);

        let swirl_speed = 10.0;
        self.current_swirl += (input.swirl_rpm - self.current_swirl) * dt * swirl_speed;

        if input.add_water {
            self.water_level = (self.water_level + 0.3).min(1.0);
        }
    }

    fn update_particles_simple(&mut self, dt: f32, shake: bool) {
        let gravity = self.effective_gravity();
        let pan_center = self.pan_center;
        let current_swirl = self.current_swirl;
        let pan_radius = self.pan_radius;
        let pan_depth = self.pan_depth;
        let mut rng = rand::thread_rng();

        for particle in self.particles.iter_mut() {
            particle.velocity += gravity * dt;

            if current_swirl > 0.1 {
                let to_center = particle.position - pan_center;
                let r = Vec2::new(to_center.x, to_center.z).length();
                if r > 0.0001 {
                    let omega = current_swirl * std::f32::consts::TAU / 60.0;
                    let tangent = Vec3::new(-to_center.z / r, 0.0, to_center.x / r);
                    let target = tangent * omega * r;
                    let blend = 1.0 - (-6.0 * dt).exp();
                    particle.velocity.x += (target.x - particle.velocity.x) * blend;
                    particle.velocity.z += (target.z - particle.velocity.z) * blend;
                }
            }

            if shake {
                let jitter = Vec3::new(
                    rng.gen_range(-0.2..0.2),
                    rng.gen_range(0.0..0.4),
                    rng.gen_range(-0.2..0.2),
                );
                particle.velocity += jitter;
            }

            particle.velocity *= 0.95;
            particle.position += particle.velocity * dt;

            let to_center = particle.position - pan_center;
            let r = Vec2::new(to_center.x, to_center.z).length();
            if r < pan_radius {
                let t = (r / pan_radius).clamp(0.0, 1.0);
                let bowl_y = pan_center.y + pan_depth * t * t;
                if particle.position.y < bowl_y {
                    particle.position.y = bowl_y;
                    if particle.velocity.y < 0.0 {
                        particle.velocity.y = 0.0;
                    }
                    particle.velocity.x *= 0.9;
                    particle.velocity.z *= 0.9;
                }
            } else if particle.position.y < pan_center.y {
                particle.position.y = pan_center.y;
                if particle.velocity.y < 0.0 {
                    particle.velocity.y = 0.0;
                }
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
        let rim_height = pan_center.y + self.pan_depth + 0.02;

        self.particles.retain(|p| {
            let delta = p.position - pan_center;
            let r = Vec2::new(delta.x, delta.z).length();
            if r <= pan_radius {
                return true;
            }
            p.position.y <= rim_height
        });
    }

    pub fn gold_remaining(&self) -> usize {
        self.particles
            .iter()
            .filter(|p| p.material == PanMaterial::Gold)
            .count()
    }

    pub fn recovery_percent(&self) -> f32 {
        if self.gold_spawned == 0 {
            return 0.0;
        }
        (self.gold_remaining() as f32 / self.gold_spawned as f32) * 100.0
    }
}
