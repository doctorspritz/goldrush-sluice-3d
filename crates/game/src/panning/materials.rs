use glam::Vec3;
use rand::Rng;

/// Material types in the pan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanMaterial {
    QuartzSand,
    Magnetite,
    Gold,
}

impl PanMaterial {
    /// Specific gravity relative to water.
    pub fn specific_gravity(&self) -> f32 {
        match self {
            PanMaterial::QuartzSand => 2.65,
            PanMaterial::Magnetite => 5.2,
            PanMaterial::Gold => 19.3,
        }
    }

    /// Particle diameter range in meters.
    pub fn size_range(&self) -> (f32, f32) {
        match self {
            PanMaterial::QuartzSand => (0.0001, 0.002),
            PanMaterial::Magnetite => (0.0001, 0.001),
            PanMaterial::Gold => (0.0002, 0.005),
        }
    }

    /// Visual color for rendering.
    pub fn color(&self) -> [f32; 3] {
        match self {
            PanMaterial::QuartzSand => [0.9, 0.85, 0.7],
            PanMaterial::Magnetite => [0.1, 0.1, 0.1],
            PanMaterial::Gold => [1.0, 0.85, 0.0],
        }
    }
}

/// A single particle in the pan.
#[derive(Clone, Copy, Debug)]
pub struct PanParticle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub material: PanMaterial,
    pub diameter: f32,
}

impl PanParticle {
    /// Create a random particle of the given material.
    pub fn random(material: PanMaterial) -> Self {
        let mut rng = rand::thread_rng();
        let (min_d, max_d) = material.size_range();

        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            material,
            diameter: rng.gen_range(min_d..max_d),
        }
    }

    pub fn specific_gravity(&self) -> f32 {
        self.material.specific_gravity()
    }

    pub fn color(&self) -> [f32; 3] {
        self.material.color()
    }
}

/// Sample to pan (collection of particles).
#[derive(Clone, Debug)]
pub struct PanSample {
    pub total_mass_grams: f32,
    pub gold_content_grams: f32,
    pub particle_count: usize,
}

impl PanSample {
    /// Tutorial sample (rich, easy).
    pub fn tutorial() -> Self {
        Self {
            total_mass_grams: 250.0,
            gold_content_grams: 10.0,
            particle_count: 2000,
        }
    }

    /// Standard sample (more realistic).
    pub fn standard() -> Self {
        Self {
            total_mass_grams: 250.0,
            gold_content_grams: 2.0,
            particle_count: 3000,
        }
    }

    /// Generate particles for this sample.
    pub fn spawn_particles(&self) -> Vec<PanParticle> {
        let mut particles = Vec::with_capacity(self.particle_count);

        let pan_center = Vec3::new(0.30, 0.02, 0.30);
        let spawn_radius = 0.16;

        let sand_count = (self.particle_count as f32 * 0.6) as usize;
        let magnetite_count = (self.particle_count as f32 * 0.25) as usize;

        for _ in 0..sand_count {
            let mut p = PanParticle::random(PanMaterial::QuartzSand);
            p.position = pan_center + Self::random_in_disk(spawn_radius);
            particles.push(p);
        }

        for _ in 0..magnetite_count {
            let mut p = PanParticle::random(PanMaterial::Magnetite);
            p.position = pan_center + Self::random_in_disk(spawn_radius);
            particles.push(p);
        }

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
        let r = rng.gen::<f32>().sqrt() * radius;

        Vec3::new(r * angle.cos(), rng.gen_range(-0.01..0.01), r * angle.sin())
    }
}
