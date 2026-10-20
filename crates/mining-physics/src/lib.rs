//! Adapter around the existing simulation, not a second fluid model. The original
//! wgpu 23 compute device owns the FLIP/APIC and heightfield solvers; Bevy's wgpu 26
//! device presents their readback. No dependency upgrade or shader rewrite is needed.
pub mod geometry;
mod terrain;
pub mod worker;
use game::gpu::{flip_3d::GpuFlip3D, heightfield::GpuHeightfield};
use glam::{Mat3, Vec3};
use mining_sim::{
    sluice::{HydraulicProfile, WaterParticle},
    GrainKind, Session, FIXED_DT,
};
use sim3d::{
    constants, ClumpShape3D, ClumpTemplate3D, ClusterSimulation3D, FlipSimulation3D, SdfParams,
};

// Original standalone demo cadence: 1 cm cells, four substeps, CFL <= 0.417.
pub const FLIP_SUBSTEPS: u32 = 4;

pub struct Hydraulics {
    device: wgpu::Device,
    queue: wgpu::Queue,
    terrain: GpuHeightfield,
    sluice: Option<LocalSluice>,
    terrain_loaded: bool,
    pub adapter_name: String,
}
struct LocalSluice {
    flip: GpuFlip3D,
    grid: sim3d::Grid3D,
    cell_types: Vec<u32>,
    dem: ClusterSimulation3D,
    slope: f32,
    profile: HydraulicProfile,
    restoring_contacts: bool,
}
impl Hydraulics {
    pub fn new() -> Result<Self, String> {
        pollster::block_on(Self::create())
    }
    async fn create() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("No GPU adapter for the original simulation")?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Original FLIP/APIC + heightfield physics"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| e.to_string())?;
        let terrain = GpuHeightfield::new(
            &device,
            mining_sim::site::WIDTH as u32,
            mining_sim::site::WIDTH as u32,
            mining_sim::site::CELL,
            0.0,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        );
        println!("PHYSICS: original FLIP/APIC + CPU DEM + GPU SWE/erosion/transport/collapse on {adapter_name}");
        Ok(Self {
            device,
            queue,
            terrain,
            sluice: None,
            terrain_loaded: false,
            adapter_name,
        })
    }
    /// Rehydrate GPU buffers from the saved authoritative channels. The renderer
    /// and camera never call this; only loading a different shift does.
    pub fn restore(&mut self) {
        self.terrain_loaded = false;
        self.sluice = None;
    }
    pub fn step(&mut self, s: &mut Session) -> Result<(), String> {
        if s.paused {
            return Ok(());
        }
        if !self.terrain_loaded {
            self.upload_terrain(s);
            self.terrain_loaded = true;
        }
        self.step_sluice(s)?;
        s.site.hydraulic.clock += FIXED_DT;
        if s.site.hydraulic.clock >= 0.0499 {
            let dt = s.site.hydraulic.clock;
            s.site.hydraulic.clock = 0.0;
            self.step_terrain(s, dt)?;
        }
        Ok(())
    }
    fn local(&mut self, s: &Session) {
        let Some(p) = s.sluice.placement else {
            self.sluice = None;
            return;
        };
        if self
            .sluice
            .as_ref()
            .is_some_and(|l| l.slope == p.slope_deg && l.profile == s.sluice.profile)
        {
            return;
        }
        let config = geometry::config(&s.sluice);
        let profile = s.sluice.profile;
        let dx = profile.cell();
        let simulation_width = config.grid_width + profile.outlet_buffer();
        let mut sim =
            FlipSimulation3D::new(simulation_width, config.grid_height, config.grid_depth, dx);
        let builder = game::sluice_geometry::SluiceGeometryBuilder::new(config.clone());
        for (i, j, k) in builder.solid_cells() {
            sim.grid.set_solid(i, j, k);
        }
        sim.grid.compute_sdf();
        let cell_types = sim
            .grid
            .solid()
            .iter()
            .map(|v| if *v { 2 } else { 0 })
            .collect();
        let mut flip = GpuFlip3D::new(
            &self.device,
            simulation_width as u32,
            config.grid_height as u32,
            config.grid_depth as u32,
            dx,
            profile.capacity() + mining_sim::sluice::MAX_GRAINS,
        );
        // CPU DEM owns solid friction. Do not add the shader's heuristic
        // low-speed damping or extra turbulence lift to those same contacts.
        flip.sediment_friction_strength = 0.0;
        // Match the standalone demo's freely entrained sediment settings;
        // contacts remain DEM-owned, rather than applying porosity drag twice.
        if profile == HydraulicProfile::OriginalDemo {
            flip.sediment_rest_particles = 0.0;
            flip.sediment_porosity_drag = 0.0;
        }
        flip.sediment_vorticity_lift = 0.0;
        flip.open_boundaries = 2 | 8; // downstream +X outlet and free surface +Y
        let dem = ClusterSimulation3D::new(
            Vec3::ZERO,
            Vec3::new(
                simulation_width as f32 * dx,
                config.grid_height as f32 * dx,
                config.grid_depth as f32 * dx,
            ),
        );
        self.sluice = Some(LocalSluice {
            flip,
            grid: sim.grid,
            cell_types,
            dem,
            slope: p.slope_deg,
            profile,
            restoring_contacts: true,
        });
    }
    fn step_sluice(&mut self, s: &mut Session) -> Result<(), String> {
        self.local(s);
        let Some(l) = &mut self.sluice else {
            return Ok(());
        };
        let profile = s.sluice.profile;
        let dx = profile.cell();
        let particle_m3 = profile.particle_m3();
        let n = (s.sluice.water_credit_m3 / particle_m3).floor() as usize;
        let emitted = n.min(profile.capacity().saturating_sub(s.sluice.water.len()));
        for i in 0..emitted {
            // Water volume uses eight particles per cell. Each new particle represents
            // exactly dx^3/8 m3 withdrawn at the river intake by Session::step.
            let seq = s
                .sluice
                .solver_steps
                .wrapping_mul(101)
                .wrapping_add(i as u64);
            let (x, y, z) = if profile == HydraulicProfile::OriginalDemo {
                // Same sheet footprint/drop as the standalone inlet. Low
                // discrepancy samples avoid repeating the coarse inlet lattice
                // while keeping the original sheet footprint deterministic.
                let sample = |factor: f64| ((seq as f64 + 0.5) * factor).fract() as f32;
                let x = (1.0 + 2.0 * sample(0.7548776662466927)) * dx;
                let y = (s.sluice.floor_cells() + 1) as f32 * dx
                    + (2.5 + 4.0 * sample(0.5698402909980532)) * dx;
                let z = profile.width() * 0.5
                    + dx
                    + (sample(0.438579021) - 0.5) * (l.grid.depth as f32 - 4.0) * dx * 0.3;
                (x, y, z)
            } else {
                (
                    0.08 + ((seq / 68) % 4) as f32 * 0.01,
                    (s.sluice.floor_cells() + 1) as f32 * dx
                        + 0.03
                        + ((seq / 272) % 3) as f32 * 0.01,
                    0.04 + (seq % 68) as f32 * 0.01,
                )
            };
            s.sluice.water.push(WaterParticle {
                position: Vec3::new(x, y, z),
                velocity: if profile == HydraulicProfile::OriginalDemo {
                    Vec3::new(0.5, -0.05, 0.0)
                } else {
                    Vec3::new(0.4, 0.0, 0.0)
                },
                affine: Mat3::ZERO,
            });
        }
        s.sluice.water_credit_m3 -= emitted as f64 * particle_m3;
        if s.sluice.water_credit_m3 > 0.02 {
            return Err(
                "FLIP capacity reached: stop the pump; water was retained, not discarded".into(),
            );
        }
        let nw = s.sluice.water.len();
        if nw + s.sluice.grains.len() == 0 {
            return Ok(());
        }
        let mut pos: Vec<_> = s
            .sluice
            .water
            .iter()
            .map(|p| p.position)
            .chain(s.sluice.grains.iter().map(|p| p.position))
            .collect();
        let mut vel: Vec<_> = s
            .sluice
            .water
            .iter()
            .map(|p| p.velocity)
            .chain(s.sluice.grains.iter().map(|p| p.velocity))
            .collect();
        let mut affine: Vec<_> = s
            .sluice
            .water
            .iter()
            .map(|p| p.affine)
            .chain(s.sluice.grains.iter().map(|p| p.affine))
            .collect();
        let density: Vec<_> = std::iter::repeat_n(1.0, nw)
            .chain(s.sluice.grains.iter().map(|g| {
                if g.kind == GrainKind::Gold {
                    constants::GOLD_DENSITY
                } else {
                    constants::GANGUE_DENSITY
                }
            }))
            .collect();
        // DEM contact history persists between steps. New parcels append clumps;
        // removal below uses the same swap operation on both arrays.
        while l.dem.clumps.len() < s.sluice.grains.len() {
            let g = &s.sluice.grains[l.dem.clumps.len()];
            let radius = g.diameter_m * 0.5;
            let density = if g.kind == GrainKind::Gold {
                constants::GOLD_DENSITY_KGM3
            } else {
                constants::GANGUE_DENSITY_KGM3
            };
            let mass = 4.0 / 3.0 * std::f32::consts::PI * radius.powi(3) * density;
            let template = ClumpTemplate3D::generate(
                ClumpShape3D::Irregular {
                    count: 1,
                    seed: 0,
                    style: sim3d::IrregularStyle3D::Round,
                },
                radius,
                mass,
            );
            let ti = l.dem.add_template(template);
            let ci = l.dem.spawn(ti, g.position, g.velocity);
            l.dem.clumps[ci].rotation = g.rotation;
            l.dem.clumps[ci].angular_velocity = g.angular_velocity;
        }
        l.dem.clumps.truncate(s.sluice.grains.len());
        if l.restoring_contacts {
            l.dem.restore_contact_history(&s.sluice.contacts);
            l.restoring_contacts = false;
        }
        let substeps = profile.substeps();
        // This slope-derived drive was present in the old demo and missing
        // from the playable adapter. Preserve its behaviour explicitly.
        let flow_accel = if profile == HydraulicProfile::OriginalDemo {
            constants::GRAVITY.abs() * (s.sluice.floor_cells() - 4) as f32 * dx / profile.length()
        } else {
            0.0
        };
        for _ in 0..substeps {
            let dt = FIXED_DT / substeps as f32;
            if !l.flip.step_async(
                &self.device,
                &self.queue,
                &mut pos,
                &mut vel,
                &mut affine,
                &density,
                &l.cell_types,
                Some(l.grid.sdf()),
                None,
                dt,
                constants::GRAVITY,
                flow_accel,
                80,
            ) {
                return Err("Original FLIP readback queue is full".into());
            }
            // The existing async slot copies positions, velocity and all three
            // APIC columns together. One fence replaces six separate downloads;
            // CPU DEM still receives the exact result of every FLIP substep.
            self.device.poll(wgpu::Maintain::Wait);
            if l.flip
                .try_readback(&self.device, &mut pos, &mut vel, &mut affine)
                != Some(pos.len())
            {
                return Err("Original FLIP readback failed".into());
            }
            for (i, c) in l.dem.clumps.iter_mut().enumerate() {
                c.position = pos[nw + i];
                c.velocity = vel[nw + i];
            }
            let sdf = SdfParams {
                sdf: l.grid.sdf(),
                grid_width: l.grid.width,
                grid_height: l.grid.height,
                grid_depth: l.grid.depth,
                cell_size: dx,
                grid_offset: Vec3::ZERO,
            };
            // Advection and gravity already happened in FLIP. Apply contacts only.
            l.dem
                .collision_response_only(dt, &sdf, !s.sluice.water.is_empty());
            for (i, c) in l.dem.clumps.iter().enumerate() {
                pos[nw + i] = c.position;
                vel[nw + i] = c.velocity;
            }
            s.sluice.solver_steps += 1;
        }
        if pos.iter().chain(&vel).any(|p| !p.is_finite()) || affine.iter().any(|c| !c.is_finite()) {
            return Err("Non-finite original solver output".into());
        }
        for (i, p) in s.sluice.water.iter_mut().enumerate() {
            p.position = pos[i];
            p.velocity = vel[i];
            p.affine = affine[i];
        }
        for (i, g) in s.sluice.grains.iter_mut().enumerate() {
            g.position = pos[nw + i];
            g.velocity = vel[nw + i];
            g.affine = affine[nw + i];
            g.rotation = l.dem.clumps[i].rotation;
            g.angular_velocity = l.dem.clumps[i].angular_velocity;
            g.resting = g.velocity.length_squared() < 0.0004;
        }
        let in_domain = |p: Vec3| {
            p.x >= 0.0
                && p.x < l.grid.width as f32 * dx
                && p.y > 0.0
                && p.y < l.grid.height as f32 * dx
                && p.z > 0.0
                && p.z < l.grid.depth as f32 * dx
        };
        let before = s.sluice.water.len();
        s.sluice.water.retain(|p| in_domain(p.position));
        let returned = (before - s.sluice.water.len()) as f64 * particle_m3;
        s.sluice.water_returned_m3 += returned;
        if let Some(outlet) = s.sluice.outlet() {
            s.site.world.add_water(outlet, returned as f32);
        }
        for i in (0..s.sluice.grains.len()).rev() {
            if !in_domain(s.sluice.grains[i].position) {
                s.sluice
                    .tailings
                    .add(s.sluice.grains.swap_remove(i).material);
                l.dem.remove_clump(i);
            }
        }
        s.sluice.contacts = l.dem.contact_history();
        // Cleanup empties the domain after the fluid drains. It never changes
        // retention probabilities or grants a target yield.
        Ok(())
    }
}
