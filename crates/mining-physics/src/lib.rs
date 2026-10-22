//! Adapter around the existing simulation, not a second fluid model. The original
//! wgpu 23 compute device owns the FLIP/APIC and heightfield solvers; Bevy's wgpu 26
//! device presents bounded snapshots. Original stages are retained, with momentum,
//! sediment coupling and GPU scheduling corrections.
pub mod calibration;
pub mod geometry;
mod particle_storage;
mod terrain;
mod terrain_gold;
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

// New granular shifts use 120 Hz FLIP/DEM, four substeps per 30 Hz fluid update.
// The 1 cm grid, pressure iterations and original FLIP/APIC solver are retained.
pub const FLIP_SUBSTEPS: u32 = 4;

#[derive(Clone, Copy, Default, Debug)]
pub struct StepTimings {
    pub fluid_ms: f64,
    pub prepare_ms: f64,
    pub dem_ms: f64,
    pub dem_sdf_ms: f64,
    pub dem_pairs_ms: f64,
    pub finish_ms: f64,
    pub terrain_ms: f64,
}

pub struct Hydraulics {
    pub timings: StepTimings,
    pub gpu_stages: std::collections::BTreeMap<String, f64>,
    /// Compute-pass device milliseconds; opt-in, excluding copies/host/readback.
    pub gpu_device_stages: std::collections::BTreeMap<String, f64>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    terrain: GpuHeightfield,
    terrain_events: game::gpu::heightfield::TerrainEventRecorder,
    sluice: Option<LocalSluice>,
    terrain_loaded: bool,
    initial_grain_storage: usize,
    pub adapter_name: String,
}
struct LocalSluice {
    flip: GpuFlip3D,
    grid: sim3d::Grid3D,
    cell_types: Vec<u32>,
    base_cell_types: Vec<u32>,
    water_cells: Vec<u32>,
    bed_fraction: Vec<f32>,
    free_templates: Vec<usize>,
    dem: ClusterSimulation3D,
    slope: f32,
    profile: HydraulicProfile,
    restoring_contacts: bool,
    riffles_enabled: bool,
    cleanups: u32,
    allocated_grains: usize,
    storage_growths: u32,
    storage_growth_encode_ms: f64,
}
impl Hydraulics {
    pub fn new() -> Result<Self, String> {
        pollster::block_on(Self::create())
    }
    /// Diagnostic reference allocation; physical admission and resolution are unchanged.
    pub fn new_with_grain_storage(grains: usize) -> Result<Self, String> {
        let mut solver = Self::new()?;
        solver.initial_grain_storage = grains;
        Ok(solver)
    }
    pub fn storage_diagnostics(&self) -> serde_json::Value {
        self.sluice.as_ref().map_or(serde_json::Value::Null, |l| serde_json::json!({
            "allocated_grains": l.allocated_grains, "allocated_particles": l.flip.particle_capacity(),
            "growths": l.storage_growths, "growth_encode_ms": l.storage_growth_encode_ms
        }))
    }
    /// Diagnostics use the same cell-centred geometry as the fluid solver.
    pub fn boundary_diagnostics(&self, s: &Session) -> serde_json::Value {
        let Some(l) = &self.sluice else {
            return serde_json::json!({});
        };
        let mut max_grain_penetration = 0.0_f32;
        let mut grain_centres_inside = 0;
        for g in &s.sluice.grains {
            let distance = l.grid.sample_sdf(g.position);
            max_grain_penetration = max_grain_penetration.max(g.collision_radius() - distance);
            grain_centres_inside += usize::from(distance < -0.002);
        }
        let water_inside = s
            .sluice
            .water
            .iter()
            .filter(|p| l.grid.sample_sdf(p.position) < -0.002)
            .count();
        serde_json::json!({"max_grain_surface_penetration_m":max_grain_penetration,
            "grain_centres_over_2mm_inside_solid":grain_centres_inside,
            "water_centres_over_2mm_inside_solid":water_inside})
    }
    /// Compare the actual registered DEM templates with canonical inventory.
    /// The old individual-grain compatibility mode is reported separately.
    pub fn material_diagnostics(&self, s: &Session) -> serde_json::Value {
        let Some(l) = &self.sluice else {
            return serde_json::json!({"ready":false});
        };
        if l.dem.clumps.len() != s.sluice.grains.len() {
            return serde_json::json!({"ready":false});
        }
        let mut expected = 0.;
        let mut registered = 0.;
        let mut count = 0;
        for (g, c) in s.sluice.grains.iter().zip(&l.dem.clumps) {
            if !g.volume_resolved {
                continue;
            }
            expected += g.material.mass_kg();
            registered += l.dem.templates[c.template_idx].mass as f64;
            count += 1;
        }
        serde_json::json!({"ready":true,"volume_resolved_grains":count,"legacy_grains":s.sluice.grains.len()-count,
            "inventory_dry_mass_kg":expected,"registered_dem_mass_kg":registered,"dem_mass_error_kg":registered-expected})
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
        let required_features = if std::env::var_os("SLUICE_PROFILE_GPU_TIMESTAMPS").is_some() {
            if std::env::var_os("SLUICE_PROFILE_STAGES").is_some() {
                return Err(
                    "Choose batched GPU timestamps or serialized stage fences, not both".into(),
                );
            }
            let required = game::gpu::timing::FEATURES;
            if !adapter.features().contains(required) {
                return Err("Adapter does not support GPU timestamp queries".into());
            }
            required
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Original FLIP/APIC + heightfield physics"),
                    required_features,
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
        let terrain_events = game::gpu::heightfield::TerrainEventRecorder::new(
            &device,
            &terrain.water,
            &terrain.geology,
        );
        println!("PHYSICS: original FLIP/APIC + CPU DEM + GPU SWE/erosion/transport/collapse on {adapter_name}");
        Ok(Self {
            timings: StepTimings::default(),
            gpu_stages: std::collections::BTreeMap::new(),
            gpu_device_stages: std::collections::BTreeMap::new(),
            device,
            queue,
            terrain,
            terrain_events,
            sluice: None,
            terrain_loaded: false,
            initial_grain_storage: mining_sim::sluice::INITIAL_GRAIN_STORAGE,
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
            s.site.hydraulic.normalize_bed_age();
            self.upload_terrain(s);
            self.terrain_loaded = true;
        }
        self.timings = StepTimings::default();
        self.gpu_stages.clear();
        self.gpu_device_stages.clear();
        let interval = if s.sluice.granular_packets {
            FIXED_DT * 2.0
        } else {
            FIXED_DT
        };
        s.sluice.fluid_clock += FIXED_DT;
        if s.sluice.fluid_clock + 1e-6 >= interval {
            s.sluice.fluid_clock = (s.sluice.fluid_clock - interval).max(0.0);
            self.step_sluice(s, interval)?;
        }
        s.site.hydraulic.clock += FIXED_DT;
        if s.site.hydraulic.clock >= 0.0499 {
            let dt = s.site.hydraulic.clock;
            s.site.hydraulic.clock = 0.0;
            let started = std::time::Instant::now();
            self.step_terrain(s, dt)?;
            self.timings.terrain_ms = started.elapsed().as_secs_f64() * 1000.0;
        }
        Ok(())
    }
    fn local(&mut self, s: &Session) -> Result<(), String> {
        let Some(p) = s.sluice.placement else {
            self.sluice = None;
            return Ok(());
        };
        if self.sluice.as_ref().is_some_and(|l| {
            l.slope == p.slope_deg
                && l.profile == s.sluice.profile
                && l.riffles_enabled == s.sluice.riffles_enabled
                && l.cleanups == s.sluice.cleanups
        }) {
            return Ok(());
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
            .collect::<Vec<_>>();
        let limits = self.device.limits();
        let allocated_grains = particle_storage::grain_capacity(
            profile.capacity(),
            s.sluice.grains.len(),
            self.initial_grain_storage,
            (limits.max_buffer_size / 5).min(limits.max_storage_buffer_binding_size as u64),
        )?;
        let mut flip = GpuFlip3D::new(
            &self.device,
            simulation_width as u32,
            config.grid_height as u32,
            config.grid_depth as u32,
            dx,
            profile.capacity() + allocated_grains,
        );
        if std::env::var_os("SLUICE_PROFILE_GPU_TIMESTAMPS").is_some() {
            flip.enable_device_timestamps(&self.device, &self.queue);
        }
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
        // The solid channel floor remains in the SDF. Its exterior must not
        // become an invisible floor under the open lip.
        flip.open_boundaries = 2 | 4 | 8; // +X outlet, -Y exterior and +Y free surface
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
            riffles_enabled: s.sluice.riffles_enabled,
            cleanups: s.sluice.cleanups,
            grid: sim.grid,
            cell_types: cell_types.clone(),
            water_cells: vec![0; cell_types.len()],
            bed_fraction: vec![0.0; cell_types.len()],
            free_templates: Vec::new(),
            base_cell_types: cell_types,
            dem,
            slope: p.slope_deg,
            profile,
            restoring_contacts: true,
            allocated_grains,
            storage_growths: 0,
            storage_growth_encode_ms: 0.,
        });
        Ok(())
    }
    fn step_sluice(&mut self, s: &mut Session, fluid_dt: f32) -> Result<(), String> {
        let prepared = std::time::Instant::now();
        // Migrate live saves containing particles beyond the old lip/buffer.
        mining_sim::outflow::release(&mut s.sluice);
        self.local(s)?;
        let Some(l) = &mut self.sluice else {
            return Ok(());
        };
        l.dem.coulomb_contacts =
            s.sluice.contact_model == mining_sim::sluice::ContactModel::Coulomb;
        l.flip.cpu_sediment_contacts = l.dem.coulomb_contacts;
        let profile = s.sluice.profile;
        let dx = profile.cell();
        let particle_m3 = profile.particle_m3();
        let n = (s.sluice.water_credit_m3 / particle_m3).floor() as usize;
        let available = profile
            .capacity()
            .saturating_sub(s.sluice.water.len() + s.sluice.outflow.len());
        let feed_emitted =
            ((s.sluice.feed_water_credit_m3 / particle_m3).floor() as usize).min(available);
        let emitted = n.min(available - feed_emitted);
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
            let id = s.sluice.allocate_water_id();
            s.sluice.water.push(WaterParticle {
                id,
                position: Vec3::new(x, y, z),
                velocity: s.sluice.inlet_velocity(),
                affine: Mat3::ZERO,
            });
        }
        s.sluice.admit_intake_water(emitted as f64 * particle_m3);
        for i in 0..feed_emitted {
            let sequence = s
                .sluice
                .solver_steps
                .wrapping_mul(101)
                .wrapping_add(i as u64);
            let position = s.sluice.feed_water_position(sequence);
            let id = s.sluice.allocate_water_id();
            s.sluice.water.push(WaterParticle {
                id,
                position,
                velocity: Vec3::new(0.4, 0., 0.),
                affine: Mat3::ZERO,
            });
        }
        s.sluice.feed_water_credit_m3 =
            (s.sluice.feed_water_credit_m3 - feed_emitted as f64 * particle_m3).max(0.);
        s.sluice.emit_intake_grains(fluid_dt);
        if s.sluice.grains.len() > l.allocated_grains {
            let limits = self.device.limits();
            let grains = particle_storage::grain_capacity(
                profile.capacity(),
                s.sluice.grains.len(),
                self.initial_grain_storage,
                (limits.max_buffer_size / 5).min(limits.max_storage_buffer_binding_size as u64),
            )?;
            let growth_started = std::time::Instant::now();
            l.flip
                .reallocate_for_upload(&self.device, &self.queue, profile.capacity() + grains)?;
            l.allocated_grains = grains;
            l.storage_growths += 1;
            l.storage_growth_encode_ms += growth_started.elapsed().as_secs_f64() * 1000.;
        }
        if s.sluice.water_credit_m3 + s.sluice.feed_water_credit_m3 > 0.02 {
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
                if g.volume_resolved {
                    g.density_kg_m3() / 1000.
                } else if g.kind == GrainKind::Gold {
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
            let radius = g.collision_radius();
            let density = if g.kind == GrainKind::Gold {
                constants::GOLD_DENSITY_KGM3
            } else {
                constants::GANGUE_DENSITY_KGM3
            };
            let mass = if g.volume_resolved {
                g.mass_kg().max(1e-12)
            } else {
                4.0 / 3.0 * std::f32::consts::PI * radius.powi(3) * density
            };
            let template = ClumpTemplate3D::generate(
                ClumpShape3D::Irregular {
                    count: 1,
                    seed: 0,
                    style: sim3d::IrregularStyle3D::Round,
                },
                radius,
                mass,
            );
            let ti = if let Some(index) = l.free_templates.pop() {
                l.dem.templates[index] = template;
                index
            } else {
                l.dem.add_template(template)
            };
            let ci = l.dem.spawn(ti, g.position, g.velocity);
            l.dem.clumps[ci].rotation = g.rotation;
            l.dem.clumps[ci].angular_velocity = g.angular_velocity;
        }
        while l.dem.clumps.len() > s.sluice.grains.len() {
            let i = l.dem.clumps.len() - 1;
            l.free_templates.push(l.dem.clumps[i].template_idx);
            l.dem.remove_clump(i);
        }
        // Volume/mass and bed feedback are independent of the physical grain
        // diameter used for drag. Legacy parcels retain their previous weights.
        let mut weights = vec![1.0; nw];
        weights.extend(s.sluice.grains.iter().map(|g| {
            if g.volume_resolved {
                (g.solid_volume() / particle_m3) as f32
            } else {
                1.0
            }
        }));
        l.flip.set_particle_volumes(weights);
        l.water_cells.fill(0);
        l.bed_fraction.fill(0.0);
        l.cell_types.copy_from_slice(&l.base_cell_types);
        let index = |p: Vec3| -> Option<usize> {
            let cell = (p / dx).floor().as_ivec3();
            if cell.min_element() < 0
                || cell.x >= l.grid.width as i32
                || cell.y >= l.grid.height as i32
                || cell.z >= l.grid.depth as i32
            {
                None
            } else {
                Some(
                    cell.z as usize * l.grid.width * l.grid.height
                        + cell.y as usize * l.grid.width
                        + cell.x as usize,
                )
            }
        };
        for water in &s.sluice.water {
            if let Some(i) = index(water.position) {
                l.water_cells[i] += 1;
            }
        }
        let mut diameters = Vec::with_capacity(s.sluice.grains.len());
        let mut wetness = Vec::with_capacity(s.sluice.grains.len());
        for g in &s.sluice.grains {
            let mut wet = false;
            for x in -1..=1 {
                for y in -1..=1 {
                    for z in -1..=1 {
                        if let Some(i) =
                            index(g.position + Vec3::new(x as f32, y as f32, z as f32) * dx)
                        {
                            wet |= l.water_cells[i] > 0;
                        }
                    }
                }
            }
            wetness.push(wet);
            diameters.push(if !g.volume_resolved {
                0.0
            } else if wet {
                g.diameter_m
            } else {
                -g.diameter_m
            });
            if g.volume_resolved && g.resting {
                let offsets = [-0.6, 0.0, 0.6];
                let samples: Vec<Vec3> = offsets
                    .into_iter()
                    .flat_map(|x| {
                        offsets
                            .into_iter()
                            .flat_map(move |y| offsets.into_iter().map(move |z| Vec3::new(x, y, z)))
                    })
                    .filter(|p| p.length_squared() <= 1.0)
                    .collect();
                for offset in &samples {
                    if let Some(i) = index(g.position + *offset * g.collision_radius()) {
                        l.bed_fraction[i] +=
                            (g.solid_volume() / samples.len() as f64 / (dx * dx * dx) as f64)
                                as f32;
                    }
                }
            }
        }
        if s.sluice.granular_packets {
            for (i, fraction) in l.bed_fraction.iter_mut().enumerate() {
                *fraction = fraction.clamp(0.0, 0.65);
                if *fraction >= 0.58 {
                    l.cell_types[i] = 2;
                }
            }
            l.flip.sediment_rest_particles = -1.0;
            l.flip.sediment_porosity_drag = 150.0 * 1e-6 / 0.003_f32.powi(2);
            l.flip
                .upload_sediment_fraction(&self.queue, &l.bed_fraction);
        }
        l.flip.upload_grain_diameters(&self.queue, nw, &diameters);
        if l.restoring_contacts {
            l.dem.restore_contact_history(&s.sluice.contacts);
            l.restoring_contacts = false;
        }
        let substeps = profile.substeps();
        // This slope-derived drive was present in the old demo and missing
        // from the playable adapter. Preserve its behaviour explicitly.
        let flow_accel = if profile == HydraulicProfile::OriginalDemo && !s.sluice.granular_packets
        {
            constants::GRAVITY.abs() * (s.sluice.floor_cells() - 4) as f32 * dx / profile.length()
        } else {
            0.0
        };
        self.timings.prepare_ms = prepared.elapsed().as_secs_f64() * 1000.0;
        for substep in 0..substeps {
            let fluid_started = std::time::Instant::now();
            let dt = fluid_dt / substeps as f32;
            let first = if substep + 1 == substeps { 0 } else { nw };
            let count = pos.len() - first;
            if count > 0 && !l.flip.prepare_particle_readback(first, count) {
                return Err("Original FLIP readback queue is full".into());
            }
            if substep == 0 {
                l.flip.step_no_readback(
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
                );
            } else {
                l.flip.upload_particle_range(
                    &self.queue,
                    nw,
                    &pos[nw..],
                    &vel[nw..],
                    &affine[nw..],
                );
                l.flip.step_in_place(
                    &self.device,
                    &self.queue,
                    pos.len() as u32,
                    &l.cell_types,
                    Some(l.grid.sdf()),
                    None,
                    dt,
                    constants::GRAVITY,
                    flow_accel,
                    80,
                );
            }
            if count > 0 {
                // Metal's blocking fence wait sleeps in 1ms increments. Poll at
                // a short sleeping interval on this sole worker, without spinning
                // or changing the required fluid/contact dependency.
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
                loop {
                    if l.flip.try_readback(
                        &self.device,
                        &mut pos[first..],
                        &mut vel[first..],
                        &mut affine[first..],
                    ) == Some(count)
                    {
                        break;
                    }
                    if std::time::Instant::now() >= deadline {
                        return Err("Original FLIP range readback timed out".into());
                    }
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
            }
            for (stage, ms) in l.flip.device_stage_timings(&self.device)? {
                *self.gpu_device_stages.entry(stage.into()).or_default() += ms;
            }
            for (stage, ms) in l.flip.stage_timings() {
                *self
                    .gpu_stages
                    .entry(format!("{stage} encoding"))
                    .or_default() += ms[0];
                *self
                    .gpu_stages
                    .entry(format!("{stage} submit_wait"))
                    .or_default() += ms[1];
            }
            self.timings.fluid_ms += fluid_started.elapsed().as_secs_f64() * 1000.0;
            let dem_started = std::time::Instant::now();
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
            l.dem.collision_response_with_wetness(dt, &sdf, &wetness);
            self.timings.dem_sdf_ms += l.dem.contact_stage_ms[0];
            self.timings.dem_pairs_ms += l.dem.contact_stage_ms[1];
            for (i, c) in l.dem.clumps.iter().enumerate() {
                pos[nw + i] = c.position;
                vel[nw + i] = c.velocity;
            }
            self.timings.dem_ms += dem_started.elapsed().as_secs_f64() * 1000.0;
            s.sluice.solver_steps += 1;
        }
        let finished = std::time::Instant::now();
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
            g.resting = g.velocity.length_squared() < 0.0009
                && l.grid.sample_sdf(g.position) <= g.collision_radius() + 0.003;
        }
        let in_domain = |p: Vec3| {
            p.x >= 0.0
                && p.x < l.grid.width as f32 * dx
                && p.y > 0.0
                && p.y < l.grid.height as f32 * dx
                && p.z > 0.0
                && p.z < l.grid.depth as f32 * dx
        };
        mining_sim::outflow::release(&mut s.sluice);
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
                l.free_templates.push(l.dem.clumps[i].template_idx);
                l.dem.remove_clump(i);
            }
        }
        s.sluice.contacts = l.dem.contact_history();
        self.timings.finish_ms = finished.elapsed().as_secs_f64() * 1000.0;
        // Cleanup empties the domain after the fluid drains. It never changes
        // retention probabilities or grants a target yield.
        Ok(())
    }
}
