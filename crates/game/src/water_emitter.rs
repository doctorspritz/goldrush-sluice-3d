use glam::{Vec2, Vec3};
use super::gpu::heightfield::GpuHeightfield;

/// Emitter for adding water to the heightfield simulation.
pub struct WaterEmitter {
    pub position: Vec3,
    pub velocity: Vec2, // Horizontal velocity (x, z)
    pub rate: f32, // Volume per second
    pub radius: f32,
    pub enabled: bool,
    
    // Sediment concentrations
    pub sediment_conc: f32,
    pub overburden_conc: f32,
    pub gravel_conc: f32,
    pub paydirt_conc: f32,
}

impl WaterEmitter {
    pub fn new(position: Vec3, rate: f32, radius: f32) -> Self {
        Self {
            position,
            velocity: Vec2::ZERO,
            rate,
            radius,
            enabled: true,
            sediment_conc: 0.0,
            overburden_conc: 0.0,
            gravel_conc: 0.0,
            paydirt_conc: 0.0,
        }
    }

    pub fn with_concentrations(
        mut self,
        sediment: f32,
        overburden: f32,
        gravel: f32,
        paydirt: f32
    ) -> Self {
        self.sediment_conc = sediment;
        self.overburden_conc = overburden;
        self.gravel_conc = gravel;
        self.paydirt_conc = paydirt;
        self
    }

    /// Update the GPU emitter state.
    pub fn update_gpu(
        &self,
        gpu_heightfield: &GpuHeightfield,
        queue: &wgpu::Queue,
        dt: f32,
    ) {
        gpu_heightfield.update_emitter(
            queue,
            self.position.x,
            self.position.z,
            self.radius,
            self.rate,
            self.sediment_conc,
            self.overburden_conc,
            self.gravel_conc,
            self.paydirt_conc,
            dt,
            self.enabled,
        );
    }

    /// Place the emitter at the cursor position, ensuring it is above the terrain.
    pub fn place_at_cursor(&mut self, hit: Vec3, hf_height: f32) {
        self.position = hit;
        self.clamp_height(hf_height);
    }

    /// Ensure the emitter is at a valid height above the terrain (e.g. 2m).
    pub fn clamp_height(&mut self, hf_height: f32) {
        // Ensure at least 5m above terrain for visibility
        self.position.y = self.position.y.max(hf_height + 5.0);
    }

    /// Generate visualization mesh data (vertices and indices) for the emitter.
    /// Returns positions and indices for a simple sphere.
    /// Resolution is the number of segments per ring.
    pub fn visualize(&self, resolution: u32) -> (Vec<Vec3>, Vec<u32>) {
        if !self.enabled {
             // Return just a small cross or something to show it exists but is disabled?
             // Or empty. Let's return a small grey diamond.
             // For now, empty if disabled to avoid clutter.
             return (Vec::new(), Vec::new());
        }

        let mut positions = Vec::new();
        let mut indices = Vec::new();

        let segments = resolution;
        let rings = resolution / 2;
        let radius = (self.radius * 0.1).max(0.5); // Visual size: 10% of actual radius
        
        // Simple sphere mesh
         for r in 0..=rings {
            let theta = std::f32::consts::PI * r as f32 / rings as f32;
            let y = radius * theta.cos();
            let ring_radius = radius * theta.sin();

            for s in 0..=segments {
                let phi = 2.0 * std::f32::consts::PI * s as f32 / segments as f32;
                let x = ring_radius * phi.cos();
                let z = ring_radius * phi.sin();

                positions.push(self.position + Vec3::new(x, y, z));
            }
        }
        
        for r in 0..rings {
            for s in 0..segments {
                let cur = r * (segments + 1) + s;
                let next = cur + segments + 1;
                
                indices.push(cur as u32);
                indices.push((cur + 1) as u32);
                indices.push(next as u32);
                
                indices.push((cur + 1) as u32);
                indices.push((next + 1) as u32);
                indices.push(next as u32);
            }
        }

        (positions, indices)
    }
}
