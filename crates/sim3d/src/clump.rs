//! 3D rigid clumps (multi-sphere clusters) for rock/gravel prototypes.

use glam::{Mat3, Quat, Vec3};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{collections::HashMap, f32::consts::PI};

#[derive(Clone, Copy, Debug)]
pub enum IrregularStyle3D {
    Round,
    Sharp,
}

#[derive(Clone, Copy, Debug)]
pub enum ClumpShape3D {
    Tetra,
    Cube2,
    Flat4,
    Rod3,
    Irregular {
        count: usize,
        seed: u64,
        style: IrregularStyle3D,
    },
}

#[derive(Clone, Debug)]
pub struct ClumpTemplate3D {
    pub shape: ClumpShape3D,
    pub local_offsets: Vec<Vec3>,
    pub particle_radius: f32,
    pub particle_mass: f32,
    pub mass: f32,
    pub inertia_inv_local: Mat3,
    pub bounding_radius: f32,
}

impl ClumpTemplate3D {
    pub fn generate(shape: ClumpShape3D, particle_radius: f32, particle_mass: f32) -> Self {
        let spacing = particle_radius * 2.2;
        let mut local_offsets = match shape {
            ClumpShape3D::Tetra => {
                let s = spacing * 0.6;
                vec![
                    Vec3::new(1.0, 1.0, 1.0) * s,
                    Vec3::new(-1.0, -1.0, 1.0) * s,
                    Vec3::new(-1.0, 1.0, -1.0) * s,
                    Vec3::new(1.0, -1.0, -1.0) * s,
                ]
            }
            ClumpShape3D::Cube2 => {
                let s = spacing * 0.75;
                let mut offsets = Vec::with_capacity(8);
                for &x in &[-1.0, 1.0] {
                    for &y in &[-1.0, 1.0] {
                        for &z in &[-1.0, 1.0] {
                            offsets.push(Vec3::new(x, y, z) * s);
                        }
                    }
                }
                offsets
            }
            ClumpShape3D::Flat4 => {
                let s = spacing;
                vec![
                    Vec3::new(-s, 0.0, -s),
                    Vec3::new(s, 0.0, -s),
                    Vec3::new(-s, 0.0, s),
                    Vec3::new(s, 0.0, s),
                ]
            }
            ClumpShape3D::Rod3 => {
                let s = spacing;
                vec![
                    Vec3::new(-s, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(s, 0.0, 0.0),
                ]
            }
            ClumpShape3D::Irregular { count, seed, style } => {
                let count = count.max(1);
                generate_irregular_offsets(count, spacing, seed, style)
            }
        };

        center_on_com(&mut local_offsets);

        let mass = particle_mass * local_offsets.len() as f32;
        let (inertia_inv_local, bounding_radius) =
            compute_inertia(&local_offsets, particle_mass, particle_radius);

        Self {
            shape,
            local_offsets,
            particle_radius,
            particle_mass,
            mass,
            inertia_inv_local,
            bounding_radius,
        }
    }

    pub fn inv_mass(&self) -> f32 {
        if self.mass <= 0.0 {
            0.0
        } else {
            1.0 / self.mass
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Clump3D {
    pub position: Vec3,
    pub velocity: Vec3,
    pub rotation: Quat,
    pub angular_velocity: Vec3,
    pub template_idx: usize,
}

impl Clump3D {
    pub fn new(position: Vec3, velocity: Vec3, template_idx: usize) -> Self {
        Self {
            position,
            velocity,
            rotation: Quat::IDENTITY,
            angular_velocity: Vec3::ZERO,
            template_idx,
        }
    }

    pub fn particle_world_position(&self, offset: Vec3) -> Vec3 {
        self.position + self.rotation * offset
    }

    pub fn particle_world_velocity(&self, offset_world: Vec3) -> Vec3 {
        self.velocity + self.angular_velocity.cross(offset_world)
    }

    pub fn world_inertia_inv(&self, template: &ClumpTemplate3D) -> Mat3 {
        let rot = Mat3::from_quat(self.rotation);
        rot * template.inertia_inv_local * rot.transpose()
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct SphereContactKey {
    a: usize,
    b: usize,
    ia: usize,
    ib: usize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct PlaneContactKey {
    clump: usize,
    particle: usize,
    axis: u8,
    side: i8,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct SdfContactKey {
    clump: usize,
    particle: usize,
}

/// Serializable contact springs for coupled solvers. Keeping these with a save
/// avoids replacing static friction by a fresh contact after restoring a shift.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ContactHistory3D {
    pub spheres: Vec<(usize, usize, usize, usize, Vec3)>,
    pub planes: Vec<(usize, usize, u8, i8, Vec3)>,
    pub sdf: Vec<(usize, usize, Vec3)>,
}

/// Parameters for SDF collision
pub struct SdfParams<'a> {
    pub sdf: &'a [f32],
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,
    pub cell_size: f32,
    /// World-space offset - subtracted from clump positions before SDF sampling
    /// Set to Vec3::ZERO if clumps are already in grid-local space
    pub grid_offset: Vec3,
}

/// Parameters for fluid coupling using Shields stress criterion.
///
/// Shields stress determines when sediment particles begin to move under fluid flow.
/// The dimensionless Shields parameter θ* = τ / ((ρs - ρf) × g × d) compares the
/// bed shear stress τ to the submerged weight of the particle.
///
/// When θ* > θ_cr (critical Shields), particles are entrained by the flow.
/// When θ* < θ_cr, particles remain stationary on the bed.
#[derive(Clone, Copy, Debug)]
pub struct FluidParams {
    /// Water density (kg/m³). Default: 1000
    pub water_density: f32,
    /// Dynamic viscosity of water (Pa·s). Default: 0.001 (water at 20°C)
    pub water_viscosity: f32,
    /// Critical Shields parameter for incipient motion. Default: 0.045
    /// Typical values: 0.03-0.06 depending on grain shape and packing
    pub critical_shields: f32,
    /// Drag coefficient for particles in flow. Default: 0.47 (sphere)
    pub drag_coefficient: f32,
    /// Lift coefficient for particles near bed. Default: 0.2
    pub lift_coefficient: f32,
    /// Bed friction coefficient for Shields calculation. Default: 0.01
    /// This relates flow velocity to bed shear stress via τ = Cf × ρf × U²
    pub bed_friction_coefficient: f32,
}

impl Default for FluidParams {
    fn default() -> Self {
        Self {
            water_density: 1000.0,
            water_viscosity: 0.001,
            critical_shields: 0.045,
            drag_coefficient: 0.47,
            lift_coefficient: 0.2,
            bed_friction_coefficient: 0.01,
        }
    }
}

/// Fluid velocity field for Shields stress calculation.
///
/// Provides the fluid velocity at a given position for calculating
/// bed shear stress and Shields parameter.
pub struct FluidVelocityField<'a> {
    /// Fluid velocity components on staggered MAC grid (U component)
    pub grid_u: &'a [f32],
    /// Fluid velocity components on staggered MAC grid (V component)
    pub grid_v: &'a [f32],
    /// Fluid velocity components on staggered MAC grid (W component)
    pub grid_w: &'a [f32],
    /// Grid dimensions
    pub grid_width: usize,
    pub grid_height: usize,
    pub grid_depth: usize,
    /// Cell size (m)
    pub cell_size: f32,
    /// Grid offset in world space
    pub grid_offset: Vec3,
}

pub struct ClusterSimulation3D {
    /// Coulomb impulses for the externally integrated fluid/contact path.
    /// False preserves the older damping-based collision behaviour.
    pub coulomb_contacts: bool,
    contact_wet: Vec<bool>,
    /// Diagnostic fallback compares the same ordered contacts using a hash map.
    pub accelerated_contacts: bool,
    pub contact_stage_ms: [f64; 2],
    contact_cells: Vec<glam::IVec3>,
    contact_pairs: Vec<(usize, usize)>,
    contact_lanes: Vec<[glam::Vec4; 4]>,
    contact_lookup: Vec<usize>,
    contact_rows: Vec<usize>,
    contact_occupied: Vec<glam::IVec3>,
    contact_members: Vec<u64>,
    contact_candidates: Vec<u64>,
    contact_inv_mass: Vec<f32>,

    pub templates: Vec<ClumpTemplate3D>,
    pub clumps: Vec<Clump3D>,
    pub gravity: Vec3,
    pub restitution: f32,
    pub friction: f32,
    pub floor_friction: f32,
    /// Friction coefficient when wet (much lower - gravel slides in water)
    pub wet_friction: f32,
    pub normal_stiffness: f32,
    pub tangential_stiffness: f32,
    pub rolling_friction: f32,
    /// Rolling friction when wet (lower)
    pub wet_rolling_friction: f32,
    pub use_dem: bool,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    sphere_contacts: HashMap<SphereContactKey, Vec3>,
    plane_contacts: HashMap<PlaneContactKey, Vec3>,
    sdf_contacts: HashMap<SdfContactKey, Vec3>,
}

impl ClusterSimulation3D {
    pub fn new(bounds_min: Vec3, bounds_max: Vec3) -> Self {
        Self {
            coulomb_contacts: false,
            contact_wet: Vec::new(),
            accelerated_contacts: true,
            contact_stage_ms: [0.; 2],
            contact_cells: Vec::new(),
            contact_pairs: Vec::new(),
            contact_lanes: Vec::new(),
            contact_inv_mass: Vec::new(),
            contact_lookup: Vec::new(),
            contact_rows: Vec::new(),
            contact_occupied: Vec::new(),
            contact_members: Vec::new(),
            contact_candidates: Vec::new(),

            templates: Vec::new(),
            clumps: Vec::new(),
            gravity: Vec3::new(0.0, -9.81, 0.0),
            restitution: 0.2,
            friction: 0.4,
            floor_friction: 0.6,
            wet_friction: 0.08, // Much lower friction when wet - gravel slides
            normal_stiffness: 6_000.0,
            tangential_stiffness: 3_000.0,
            rolling_friction: 0.02,
            wet_rolling_friction: 0.002, // Very low rolling friction when wet
            use_dem: true,
            bounds_min,
            bounds_max,
            sphere_contacts: HashMap::new(),
            plane_contacts: HashMap::new(),
            sdf_contacts: HashMap::new(),
        }
    }

    pub fn add_template(&mut self, template: ClumpTemplate3D) -> usize {
        self.templates.push(template);
        self.templates.len() - 1
    }

    pub fn spawn(&mut self, template_idx: usize, position: Vec3, velocity: Vec3) -> usize {
        let clump = Clump3D::new(position, velocity, template_idx);
        self.clumps.push(clump);
        self.clumps.len() - 1
    }

    /// Returns the number of sphere-sphere contacts detected in the last DEM step
    pub fn sphere_contact_count(&self) -> usize {
        self.sphere_contacts.len()
    }

    /// Returns the number of plane contacts detected in the last DEM step
    pub fn plane_contact_count(&self) -> usize {
        self.plane_contacts.len()
    }

    /// Checks if two clumps have a sphere-sphere contact (order-independent)
    pub fn has_contact(&self, i: usize, j: usize) -> bool {
        self.sphere_contacts
            .keys()
            .any(|k| (k.a == i && k.b == j) || (k.a == j && k.b == i))
    }

    pub fn contact_history(&self) -> ContactHistory3D {
        ContactHistory3D {
            spheres: self
                .sphere_contacts
                .iter()
                .map(|(k, v)| (k.a, k.b, k.ia, k.ib, *v))
                .collect(),
            planes: self
                .plane_contacts
                .iter()
                .map(|(k, v)| (k.clump, k.particle, k.axis, k.side, *v))
                .collect(),
            sdf: self
                .sdf_contacts
                .iter()
                .map(|(k, v)| (k.clump, k.particle, *v))
                .collect(),
        }
    }

    pub fn restore_contact_history(&mut self, state: &ContactHistory3D) {
        self.sphere_contacts = state
            .spheres
            .iter()
            .map(|&(a, b, ia, ib, v)| (SphereContactKey { a, b, ia, ib }, v))
            .collect();
        self.plane_contacts = state
            .planes
            .iter()
            .map(|&(clump, particle, axis, side, v)| {
                (
                    PlaneContactKey {
                        clump,
                        particle,
                        axis,
                        side,
                    },
                    v,
                )
            })
            .collect();
        self.sdf_contacts = state
            .sdf
            .iter()
            .map(|&(clump, particle, v)| (SdfContactKey { clump, particle }, v))
            .collect();
    }

    /// Remove an exited grain without assigning its contact springs to the
    /// grain swapped in from the end of the array.
    pub fn remove_clump(&mut self, index: usize) {
        let last = self.clumps.len() - 1;
        self.clumps.swap_remove(index);
        let remap = |i: usize| if i == last { index } else { i };
        self.sphere_contacts = self
            .sphere_contacts
            .drain()
            .filter_map(|(mut k, v)| {
                if k.a == index || k.b == index {
                    return None;
                }
                k.a = remap(k.a);
                k.b = remap(k.b);
                Some((k, v))
            })
            .collect();
        self.plane_contacts = self
            .plane_contacts
            .drain()
            .filter_map(|(mut k, v)| {
                if k.clump == index {
                    return None;
                }
                k.clump = remap(k.clump);
                Some((k, v))
            })
            .collect();
        self.sdf_contacts = self
            .sdf_contacts
            .drain()
            .filter_map(|(mut k, v)| {
                if k.clump == index {
                    return None;
                }
                k.clump = remap(k.clump);
                Some((k, v))
            })
            .collect();
    }

    pub fn step(&mut self, dt: f32) {
        // Use substeps for stability. With stiffness k=6000 and mass m=0.025kg,
        // critical timestep is ~0.004s. Using 4 substeps gives sub_dt ~0.004s.
        let substeps = 8;
        let sub_dt = dt / substeps as f32;

        for _ in 0..substeps {
            if self.use_dem {
                self.step_dem_internal(sub_dt, None);
            } else {
                for clump in &mut self.clumps {
                    clump.velocity += self.gravity * sub_dt;
                    clump.position += clump.velocity * sub_dt;
                    let delta = Quat::from_scaled_axis(clump.angular_velocity * sub_dt);
                    clump.rotation = (delta * clump.rotation).normalize();
                }

                self.resolve_bounds(sub_dt);
                self.resolve_clump_contacts();
            }
        }
    }

    /// Step with SDF collision - particles collide with solid geometry defined by SDF
    pub fn step_with_sdf(&mut self, dt: f32, sdf_params: &SdfParams) {
        // Use substeps for stability. With stiffness k=6000 and mass m=0.025kg,
        // critical timestep is ~0.004s. Using 4 substeps gives sub_dt ~0.004s.
        // critical timestep is ~0.004s. Using 4 substeps gives sub_dt ~0.004s.
        // INTEGRATION UPDATE: Increased to 20 substeps for stability with irregular clumps (sand)
        let substeps = 20;
        let sub_dt = dt / substeps as f32;

        for _ in 0..substeps {
            if self.use_dem {
                self.step_dem_internal(sub_dt, Some(sdf_params));
            } else {
                for clump in &mut self.clumps {
                    clump.velocity += self.gravity * sub_dt;
                    clump.position += clump.velocity * sub_dt;
                    let delta = Quat::from_scaled_axis(clump.angular_velocity * sub_dt);
                    clump.rotation = (delta * clump.rotation).normalize();
                }

                self.resolve_bounds(sub_dt);
                self.resolve_clump_contacts();
            }
        }
    }

    /// Collision response only - for use with FLIP coupling.
    ///
    /// This method DOES NOT integrate velocity→position (FLIP already moved particles).
    /// It only:
    /// 1. Detects SDF penetrations
    /// 2. Pushes particles out of solids
    /// 3. Applies velocity corrections (bounce, friction)
    /// 4. Updates rotation from angular velocity
    ///
    /// Use this when DEM is coupled with FLIP and FLIP handles advection.
    ///
    /// # Arguments
    /// * `wet` - If true, uses wet friction (much lower) for sliding gravel in water
    pub fn collision_response_only(&mut self, dt: f32, sdf_params: &SdfParams, wet: bool) {
        self.collision_response_local(dt, sdf_params, |_| wet);
    }

    /// Coupled callers supply local immersion, rather than wetting the entire
    /// channel merely because water exists somewhere in it.
    pub fn collision_response_with_wetness(
        &mut self,
        dt: f32,
        sdf_params: &SdfParams,
        wet: &[bool],
    ) {
        assert_eq!(wet.len(), self.clumps.len());
        self.collision_response_local(dt, sdf_params, |i| wet[i]);
    }

    fn collision_response_local(
        &mut self,
        dt: f32,
        sdf_params: &SdfParams,
        wet: impl Fn(usize) -> bool,
    ) {
        if self.clumps.is_empty() {
            return;
        }
        if self.coulomb_contacts {
            self.contact_wet.clear();
            self.contact_wet.extend((0..self.clumps.len()).map(&wet));
        }

        let mut new_sdf_contacts: HashMap<SdfContactKey, Vec3> = HashMap::new();

        let contact_started = std::time::Instant::now();
        // SDF collision detection and response
        for (idx, clump) in self.clumps.iter_mut().enumerate() {
            let friction = if wet(idx) {
                self.wet_friction
            } else {
                self.floor_friction
            };
            let rolling_friction = if wet(idx) {
                self.wet_rolling_friction
            } else {
                self.rolling_friction
            };

            let template = &self.templates[clump.template_idx];

            for (p_idx, offset) in template.local_offsets.iter().enumerate() {
                let r = clump.rotation * *offset;
                let pos = clump.position + r;
                let vel = clump.velocity + clump.angular_velocity.cross(r);
                let radius = template.particle_radius;

                // Sample SDF at particle position
                let (sdf_value, sdf_normal) = sample_sdf_contact(sdf_params, pos, radius);

                // Check for penetration: SDF < radius means sphere penetrates solid
                let penetration = radius - sdf_value;
                if penetration > 0.0 && sdf_normal.length_squared() > 1e-6 {
                    let normal = sdf_normal.normalize();

                    // Push particle out of solid (position correction)
                    clump.position += normal * penetration * 1.01; // 1% extra to avoid re-penetration
                    if self.coulomb_contacts {
                        crate::contact::resolve_static(
                            clump,
                            template,
                            normal,
                            r - normal * radius,
                            crate::contact::ContactMaterial {
                                friction,
                                restitution: self.restitution,
                                rolling: rolling_friction,
                            },
                        );
                        new_sdf_contacts.insert(
                            SdfContactKey {
                                clump: idx,
                                particle: p_idx,
                            },
                            Vec3::ZERO,
                        );
                        continue;
                    }

                    // Velocity correction - remove velocity into solid + apply friction
                    let v_n = vel.dot(normal);
                    if v_n < 0.0 {
                        // Moving into solid - bounce with restitution
                        let v_normal = normal * v_n;
                        let v_tangent = vel - v_normal;

                        // Apply restitution to normal component
                        let new_v_normal = -v_normal * self.restitution;

                        // Apply friction to tangent component (wet = low friction = slides easily)
                        // Use exponential decay for frame-rate independence: exp(-rate*dt) instead of 1-rate*dt
                        let friction_damp = (-friction * 10.0 * dt).exp();
                        let new_v_tangent = v_tangent * friction_damp;

                        // Reconstruct velocity (without angular contribution for now)
                        clump.velocity = new_v_normal + new_v_tangent;

                        // Track contact for history-based friction
                        let key = SdfContactKey {
                            clump: idx,
                            particle: p_idx,
                        };
                        let prev = self.sdf_contacts.get(&key).copied().unwrap_or(Vec3::ZERO);
                        let delta_t = prev + v_tangent * dt;
                        new_sdf_contacts.insert(key, delta_t);

                        // Apply rolling friction to angular velocity
                        // Use exponential decay for frame-rate independence
                        // Rate 120.0 preserves original behavior tuned for ~60Hz (2.0 / (1/60) = 120)
                        if rolling_friction > 0.0 && clump.angular_velocity.length_squared() > 1e-8
                        {
                            let roll_damp = (-rolling_friction * 120.0 * dt).exp();
                            clump.angular_velocity *= roll_damp;
                        }
                    }
                }
            }

            // Update rotation from angular velocity
            if clump.angular_velocity.length_squared() > 1e-10 {
                let delta = Quat::from_scaled_axis(clump.angular_velocity * dt);
                clump.rotation = (delta * clump.rotation).normalize();
            }
        }

        self.contact_stage_ms[0] = contact_started.elapsed().as_secs_f64() * 1000.0;
        let pairs_started = std::time::Instant::now();
        // Inter-clump collision (simple position-based resolution)
        self.resolve_dem_penetrations(2);
        self.contact_stage_ms[1] = pairs_started.elapsed().as_secs_f64() * 1000.0;

        self.sdf_contacts = new_sdf_contacts;
    }

    /// Compute sphere-sphere collision forces using spatial hashing for O(n) detection.
    fn compute_sphere_sphere_forces(
        &self,
        dt: f32,
        forces: &mut [Vec3],
        torques: &mut [Vec3],
    ) -> HashMap<SphereContactKey, Vec3> {
        let mut new_sphere_contacts: HashMap<SphereContactKey, Vec3> = HashMap::new();

        // Spatial hashing for O(n) collision detection
        // Cell size = 2x max bounding radius so neighbors are always in adjacent cells
        let max_radius = self
            .templates
            .iter()
            .map(|t| t.bounding_radius)
            .fold(0.0f32, f32::max);
        let cell_size = max_radius * 2.0 + 0.001; // Small epsilon to avoid edge cases
        let inv_cell_size = 1.0 / cell_size;

        // Build spatial hash: cell -> list of clump indices
        let mut spatial_hash: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (idx, clump) in self.clumps.iter().enumerate() {
            let cx = (clump.position.x * inv_cell_size).floor() as i32;
            let cy = (clump.position.y * inv_cell_size).floor() as i32;
            let cz = (clump.position.z * inv_cell_size).floor() as i32;
            spatial_hash.entry((cx, cy, cz)).or_default().push(idx);
        }

        // Check collisions only with neighbors (27 cells including self)
        let mut checked_pairs: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();

        for (&(cx, cy, cz), indices) in &spatial_hash {
            for &i in indices {
                // Check all 27 neighboring cells (including self)
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let neighbor_key = (
                                cx.saturating_add(dx),
                                cy.saturating_add(dy),
                                cz.saturating_add(dz),
                            );
                            if let Some(neighbor_indices) = spatial_hash.get(&neighbor_key) {
                                for &j in neighbor_indices {
                                    // Skip self and already-checked pairs
                                    if i >= j {
                                        continue;
                                    }
                                    // Avoid duplicate checks
                                    let pair = (i, j);
                                    if checked_pairs.contains(&pair) {
                                        continue;
                                    }
                                    checked_pairs.insert(pair);

                                    self.compute_clump_pair_forces(
                                        i,
                                        j,
                                        dt,
                                        forces,
                                        torques,
                                        &mut new_sphere_contacts,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        new_sphere_contacts
    }

    /// Compute forces between a pair of clumps.
    fn compute_clump_pair_forces(
        &self,
        i: usize,
        j: usize,
        dt: f32,
        forces: &mut [Vec3],
        torques: &mut [Vec3],
        new_sphere_contacts: &mut HashMap<SphereContactKey, Vec3>,
    ) {
        let clump_a = &self.clumps[i];
        let clump_b = &self.clumps[j];
        let template_a = &self.templates[clump_a.template_idx];
        let template_b = &self.templates[clump_b.template_idx];

        let delta = clump_b.position - clump_a.position;
        let max_dist = template_a.bounding_radius + template_b.bounding_radius;
        if delta.length_squared() > max_dist * max_dist {
            return;
        }

        let contact_dist = template_a.particle_radius + template_b.particle_radius;
        let contact_dist_sq = contact_dist * contact_dist;
        let m_eff = (template_a.particle_mass * template_b.particle_mass)
            / (template_a.particle_mass + template_b.particle_mass);

        let c_n = dem_damping(self.restitution, self.normal_stiffness, m_eff);
        let c_t = dem_damping(self.restitution, self.tangential_stiffness, m_eff);

        for (ia, offset_a) in template_a.local_offsets.iter().enumerate() {
            let ra = clump_a.rotation * *offset_a;
            let pa = clump_a.position + ra;
            let va = clump_a.velocity + clump_a.angular_velocity.cross(ra);

            for (ib, offset_b) in template_b.local_offsets.iter().enumerate() {
                let rb = clump_b.rotation * *offset_b;
                let pb = clump_b.position + rb;
                let vb = clump_b.velocity + clump_b.angular_velocity.cross(rb);

                let diff = pb - pa;
                let dist_sq = diff.length_squared();
                if dist_sq >= contact_dist_sq {
                    continue;
                }

                let (normal, dist) = if dist_sq > 1.0e-10 {
                    let dist = dist_sq.sqrt();
                    (diff / dist, dist)
                } else {
                    let rel = vb - va;
                    let fallback = if rel.length_squared() > 1.0e-10 {
                        rel.normalize()
                    } else {
                        let center = clump_b.position - clump_a.position;
                        if center.length_squared() > 1.0e-10 {
                            center.normalize()
                        } else {
                            Vec3::Y
                        }
                    };
                    (fallback, 0.0)
                };
                let penetration = contact_dist - dist;
                if penetration <= 0.0 {
                    continue;
                }

                let rel_vel = vb - va;
                let v_n = rel_vel.dot(normal);
                let mut fn_mag = self.normal_stiffness * penetration - c_n * v_n;
                if fn_mag < 0.0 {
                    fn_mag = 0.0;
                }

                let vt = rel_vel - normal * v_n;
                let key = SphereContactKey { a: i, b: j, ia, ib };
                let prev = self
                    .sphere_contacts
                    .get(&key)
                    .copied()
                    .unwrap_or(Vec3::ZERO);
                let mut delta_t = prev + vt * dt;

                let mut ft = -self.tangential_stiffness * delta_t - c_t * vt;
                let max_ft = self.friction * fn_mag;
                if ft.length_squared() > max_ft * max_ft {
                    if ft.length_squared() > 1.0e-10 {
                        ft = ft.normalize() * max_ft;
                    } else {
                        ft = Vec3::ZERO;
                    }
                    if self.tangential_stiffness > 0.0 {
                        delta_t = -(ft + c_t * vt) / self.tangential_stiffness;
                    }
                }
                new_sphere_contacts.insert(key, delta_t);

                let total = normal * fn_mag + ft;
                forces[i] -= total;
                forces[j] += total;
                torques[i] += ra.cross(-total);
                torques[j] += rb.cross(total);

                if self.rolling_friction > 0.0 {
                    if clump_a.angular_velocity.length_squared() > 1.0e-8 {
                        let roll = -clump_a.angular_velocity.normalize()
                            * (self.rolling_friction * fn_mag * template_a.particle_radius);
                        torques[i] += roll;
                    }
                    if clump_b.angular_velocity.length_squared() > 1.0e-8 {
                        let roll = -clump_b.angular_velocity.normalize()
                            * (self.rolling_friction * fn_mag * template_b.particle_radius);
                        torques[j] += roll;
                    }
                }
            }
        }
    }

    /// Compute boundary/plane collision forces for all particles against world bounds.
    fn compute_boundary_forces(
        &self,
        dt: f32,
        forces: &mut [Vec3],
        torques: &mut [Vec3],
    ) -> HashMap<PlaneContactKey, Vec3> {
        let mut new_plane_contacts: HashMap<PlaneContactKey, Vec3> = HashMap::new();

        for (idx, clump) in self.clumps.iter().enumerate() {
            let template = &self.templates[clump.template_idx];
            for (p_idx, offset) in template.local_offsets.iter().enumerate() {
                let r = clump.rotation * *offset;
                let pos = clump.position + r;
                let vel = clump.velocity + clump.angular_velocity.cross(r);
                let radius = template.particle_radius;

                // X-axis bounds
                if pos.x - radius < self.bounds_min.x {
                    let penetration = self.bounds_min.x - (pos.x - radius);
                    self.apply_plane_contact(
                        idx,
                        p_idx,
                        0,
                        -1,
                        Vec3::X,
                        penetration,
                        r,
                        vel,
                        template.particle_mass,
                        radius,
                        clump.angular_velocity,
                        dt,
                        forces,
                        torques,
                        &mut new_plane_contacts,
                    );
                } else if pos.x + radius > self.bounds_max.x {
                    let penetration = pos.x + radius - self.bounds_max.x;
                    self.apply_plane_contact(
                        idx,
                        p_idx,
                        0,
                        1,
                        -Vec3::X,
                        penetration,
                        r,
                        vel,
                        template.particle_mass,
                        radius,
                        clump.angular_velocity,
                        dt,
                        forces,
                        torques,
                        &mut new_plane_contacts,
                    );
                }

                // Y-axis bounds
                if pos.y - radius < self.bounds_min.y {
                    let penetration = self.bounds_min.y - (pos.y - radius);
                    self.apply_plane_contact(
                        idx,
                        p_idx,
                        1,
                        -1,
                        Vec3::Y,
                        penetration,
                        r,
                        vel,
                        template.particle_mass,
                        radius,
                        clump.angular_velocity,
                        dt,
                        forces,
                        torques,
                        &mut new_plane_contacts,
                    );
                } else if pos.y + radius > self.bounds_max.y {
                    let penetration = pos.y + radius - self.bounds_max.y;
                    self.apply_plane_contact(
                        idx,
                        p_idx,
                        1,
                        1,
                        -Vec3::Y,
                        penetration,
                        r,
                        vel,
                        template.particle_mass,
                        radius,
                        clump.angular_velocity,
                        dt,
                        forces,
                        torques,
                        &mut new_plane_contacts,
                    );
                }

                // Z-axis bounds
                if pos.z - radius < self.bounds_min.z {
                    let penetration = self.bounds_min.z - (pos.z - radius);
                    self.apply_plane_contact(
                        idx,
                        p_idx,
                        2,
                        -1,
                        Vec3::Z,
                        penetration,
                        r,
                        vel,
                        template.particle_mass,
                        radius,
                        clump.angular_velocity,
                        dt,
                        forces,
                        torques,
                        &mut new_plane_contacts,
                    );
                } else if pos.z + radius > self.bounds_max.z {
                    let penetration = pos.z + radius - self.bounds_max.z;
                    self.apply_plane_contact(
                        idx,
                        p_idx,
                        2,
                        1,
                        -Vec3::Z,
                        penetration,
                        r,
                        vel,
                        template.particle_mass,
                        radius,
                        clump.angular_velocity,
                        dt,
                        forces,
                        torques,
                        &mut new_plane_contacts,
                    );
                }
            }
        }

        new_plane_contacts
    }

    /// Compute SDF collision forces - particles collide with solid geometry defined by SDF.
    fn compute_sdf_forces(
        &self,
        sdf: &SdfParams,
        dt: f32,
        forces: &mut [Vec3],
        torques: &mut [Vec3],
    ) -> HashMap<SdfContactKey, Vec3> {
        let mut new_sdf_contacts: HashMap<SdfContactKey, Vec3> = HashMap::new();

        for (idx, clump) in self.clumps.iter().enumerate() {
            let template = &self.templates[clump.template_idx];
            for (p_idx, offset) in template.local_offsets.iter().enumerate() {
                let r = clump.rotation * *offset;
                let pos = clump.position + r;
                let vel = clump.velocity + clump.angular_velocity.cross(r);
                let radius = template.particle_radius;

                // Sample SDF at particle position
                let (sdf_value, sdf_normal) = sample_sdf_with_gradient(
                    sdf.sdf,
                    pos,
                    sdf.grid_offset,
                    sdf.grid_width,
                    sdf.grid_height,
                    sdf.grid_depth,
                    sdf.cell_size,
                );

                // Check for penetration: SDF < radius means sphere penetrates solid
                let mut penetration = radius - sdf_value;
                if penetration > 0.0 && sdf_normal.length_squared() > 1e-6 {
                    let normal = sdf_normal.normalize();

                    // Clamp penetration depth to prevent force explosion
                    // Deep penetrations cause astronomical forces that make particles explode
                    // Limit to 50% of particle radius (for 8mm particles, max 2mm penetration)
                    let max_penetration = radius * 0.5;
                    penetration = penetration.min(max_penetration);

                    // Apply contact force using spring-damper model
                    let v_n = vel.dot(normal);
                    let c_n = dem_damping(
                        self.restitution,
                        self.normal_stiffness,
                        template.particle_mass,
                    );
                    let c_t = dem_damping(
                        self.restitution,
                        self.tangential_stiffness,
                        template.particle_mass,
                    );

                    let mut fn_mag = self.normal_stiffness * penetration - c_n * v_n;
                    if fn_mag < 0.0 {
                        fn_mag = 0.0;
                    }

                    // Clamp force to prevent numerical explosion from deep penetrations
                    // Max force: particle weight * 1000 (reasonable for particle collisions)
                    let max_force = template.particle_mass * self.gravity.length() * 1000.0;
                    fn_mag = fn_mag.min(max_force);

                    // Tangential (friction) force
                    let vt = vel - normal * v_n;
                    let key = SdfContactKey {
                        clump: idx,
                        particle: p_idx,
                    };
                    let prev = self.sdf_contacts.get(&key).copied().unwrap_or(Vec3::ZERO);
                    let mut delta_t = prev + vt * dt;

                    let mut ft = -self.tangential_stiffness * delta_t - c_t * vt;
                    let max_ft = self.floor_friction * fn_mag; // Use floor friction for SDF solids
                    if ft.length_squared() > max_ft * max_ft {
                        if ft.length_squared() > 1.0e-10 {
                            ft = ft.normalize() * max_ft;
                        } else {
                            ft = Vec3::ZERO;
                        }
                        if self.tangential_stiffness > 0.0 {
                            delta_t = -(ft + c_t * vt) / self.tangential_stiffness;
                        }
                    }
                    new_sdf_contacts.insert(key, delta_t);

                    let total = normal * fn_mag + ft;
                    forces[idx] += total;
                    torques[idx] += r.cross(total);

                    // Rolling friction
                    if self.rolling_friction > 0.0
                        && clump.angular_velocity.length_squared() > 1.0e-8
                    {
                        let roll = -clump.angular_velocity.normalize()
                            * (self.rolling_friction * fn_mag * radius);
                        torques[idx] += roll;
                    }
                }
            }
        }

        new_sdf_contacts
    }

    /// Integrate forces and torques into velocity and position.
    fn integrate_motion(&mut self, dt: f32, forces: &[Vec3], torques: &[Vec3]) {
        for (idx, clump) in self.clumps.iter_mut().enumerate() {
            let template = &self.templates[clump.template_idx];
            let inv_mass = template.inv_mass();
            if inv_mass == 0.0 {
                continue;
            }

            let accel = forces[idx] * inv_mass;
            clump.velocity += accel * dt;
            clump.position += clump.velocity * dt;

            let inv_inertia = clump.world_inertia_inv(template);
            let angular_accel = inv_inertia * torques[idx];
            clump.angular_velocity += angular_accel * dt;
            let delta = Quat::from_scaled_axis(clump.angular_velocity * dt);
            clump.rotation = (delta * clump.rotation).normalize();
        }
    }

    /// Safety clamp: fix deep SDF penetrations and limit max speed to avoid runaway forces.
    fn clamp_sdf_penetration(&mut self, sdf: &SdfParams) {
        let max_speed = 50.0;
        for clump in &mut self.clumps {
            let template = &self.templates[clump.template_idx];
            // Allow tiny numerical penetration (1% of radius) but no more
            let max_penetration = template.particle_radius * 0.01;
            for offset in &template.local_offsets {
                let r = clump.rotation * *offset;
                let pos = clump.position + r;
                let (sdf_value, sdf_normal) = sample_sdf_with_gradient(
                    sdf.sdf,
                    pos,
                    sdf.grid_offset,
                    sdf.grid_width,
                    sdf.grid_height,
                    sdf.grid_depth,
                    sdf.cell_size,
                );
                let penetration = template.particle_radius - sdf_value;
                if penetration > max_penetration && sdf_normal.length_squared() > 1e-6 {
                    let normal = sdf_normal.normalize();
                    let excess = penetration - max_penetration;
                    clump.position += normal * excess;
                    let v_n = clump.velocity.dot(normal);
                    if v_n < 0.0 {
                        clump.velocity -= normal * v_n;
                    }
                    // Dampen angular velocity to prevent drilling back in
                    clump.angular_velocity *= 0.5;
                }
            }
            let speed_sq = clump.velocity.length_squared();
            if speed_sq > max_speed * max_speed {
                clump.velocity = clump.velocity.normalize() * max_speed;
            }
        }
    }

    /// Prevent deep SDF penetration BEFORE force calculation.
    /// This stops particles from getting embedded deep in geometry, which causes
    /// force explosion even with clamped penetration depths.
    fn prevent_deep_sdf_penetration(&mut self, sdf: &SdfParams) {
        for clump in &mut self.clumps {
            let template = &self.templates[clump.template_idx];
            let max_allowed = template.particle_radius * 0.1; // Allow only 10% penetration

            for offset in &template.local_offsets {
                let r = clump.rotation * *offset;
                let pos = clump.position + r;
                let (sdf_value, sdf_normal) = sample_sdf_with_gradient(
                    sdf.sdf,
                    pos,
                    sdf.grid_offset,
                    sdf.grid_width,
                    sdf.grid_height,
                    sdf.grid_depth,
                    sdf.cell_size,
                );

                let penetration = template.particle_radius - sdf_value;
                if penetration > max_allowed && sdf_normal.length_squared() > 1e-6 {
                    let excess = penetration - max_allowed;
                    let normal = sdf_normal.normalize();
                    clump.position += normal * excess;

                    // Zero velocity into surface
                    let v_n = clump.velocity.dot(normal);
                    if v_n < 0.0 {
                        clump.velocity -= normal * v_n;
                    }

                    // Dampen angular velocity to prevent "drilling".
                    // Rotation can drive particles deep into the floor. If we only fix position
                    // but leave angular velocity, it just rotates back in, gaining energy from the
                    // position correction (potential energy pump).
                    clump.angular_velocity *= 0.5;
                }
            }
        }
    }

    fn step_dem_internal(&mut self, dt: f32, sdf_params: Option<&SdfParams>) {
        if self.clumps.is_empty() {
            return;
        }

        if let Some(sdf) = sdf_params {
            self.prevent_deep_sdf_penetration(sdf);
        }

        let mut forces = vec![Vec3::ZERO; self.clumps.len()];
        let mut torques = vec![Vec3::ZERO; self.clumps.len()];

        for (idx, clump) in self.clumps.iter().enumerate() {
            let template = &self.templates[clump.template_idx];
            forces[idx] += self.gravity * template.mass;
        }

        let new_sphere_contacts = self.compute_sphere_sphere_forces(dt, &mut forces, &mut torques);

        let new_plane_contacts = self.compute_boundary_forces(dt, &mut forces, &mut torques);

        let new_sdf_contacts = if let Some(sdf) = sdf_params {
            self.compute_sdf_forces(sdf, dt, &mut forces, &mut torques)
        } else {
            HashMap::new()
        };

        self.integrate_motion(dt, &forces, &torques);

        if let Some(sdf) = sdf_params {
            self.clamp_sdf_penetration(sdf);
        }

        self.sphere_contacts = new_sphere_contacts;
        self.plane_contacts = new_plane_contacts;
        self.sdf_contacts = new_sdf_contacts;
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_plane_contact(
        &self,
        clump_idx: usize,
        particle_idx: usize,
        axis: u8,
        side: i8,
        normal: Vec3,
        penetration: f32,
        r: Vec3,
        vel: Vec3,
        particle_mass: f32,
        particle_radius: f32,
        angular_velocity: Vec3,
        dt: f32,
        forces: &mut [Vec3],
        torques: &mut [Vec3],
        new_plane_contacts: &mut HashMap<PlaneContactKey, Vec3>,
    ) {
        if penetration <= 0.0 {
            return;
        }

        let v_n = vel.dot(normal);
        let c_n = dem_damping(self.restitution, self.normal_stiffness, particle_mass);
        let c_t = dem_damping(self.restitution, self.tangential_stiffness, particle_mass);
        let mut fn_mag = self.normal_stiffness * penetration - c_n * v_n;
        if fn_mag < 0.0 {
            fn_mag = 0.0;
        }

        let vt = vel - normal * v_n;
        let key = PlaneContactKey {
            clump: clump_idx,
            particle: particle_idx,
            axis,
            side,
        };
        let prev = self.plane_contacts.get(&key).copied().unwrap_or(Vec3::ZERO);
        let mut delta_t = prev + vt * dt;

        let mut ft = -self.tangential_stiffness * delta_t - c_t * vt;
        let friction = if axis == 1 && side == -1 {
            self.floor_friction
        } else {
            self.friction
        };
        let max_ft = friction * fn_mag;
        if ft.length_squared() > max_ft * max_ft {
            if ft.length_squared() > 1.0e-10 {
                ft = ft.normalize() * max_ft;
            } else {
                ft = Vec3::ZERO;
            }
            if self.tangential_stiffness > 0.0 {
                delta_t = -(ft + c_t * vt) / self.tangential_stiffness;
            }
        }
        new_plane_contacts.insert(key, delta_t);

        let total = normal * fn_mag + ft;
        forces[clump_idx] += total;
        torques[clump_idx] += r.cross(total);

        // Rolling friction torque (same pattern as SDF and clump-clump contacts)
        if self.rolling_friction > 0.0 && angular_velocity.length_squared() > 1.0e-8 {
            let roll_torque =
                -angular_velocity.normalize() * (self.rolling_friction * fn_mag * particle_radius);
            torques[clump_idx] += roll_torque;
        }
    }

    fn resolve_dem_penetrations(&mut self, iterations: usize) {
        let percent = 0.75;
        let count = self.clumps.len();

        self.contact_inv_mass.clear();
        self.contact_inv_mass
            .extend(self.templates.iter().map(ClumpTemplate3D::inv_mass));
        let mut pairs = std::mem::take(&mut self.contact_pairs);
        for _ in 0..iterations {
            pairs.clear();
            if count <= 64 {
                for i in 0..count {
                    for j in i + 1..count {
                        pairs.push((i, j));
                    }
                }
            } else {
                let diameter = self
                    .templates
                    .iter()
                    .map(|t| t.bounding_radius * 2.0)
                    .fold(0.001_f32, f32::max);
                self.contact_cells.clear();
                self.contact_cells.extend(
                    self.clumps
                        .iter()
                        .map(|c| (c.position / diameter).floor().as_ivec3()),
                );
                let min_cell = self
                    .contact_cells
                    .iter()
                    .copied()
                    .reduce(|a, b| a.min(b))
                    .unwrap_or_default()
                    - glam::IVec3::ONE;
                let max_cell = self
                    .contact_cells
                    .iter()
                    .copied()
                    .reduce(|a, b| a.max(b))
                    .unwrap_or_default()
                    + glam::IVec3::ONE;
                let dims = max_cell - min_cell + glam::IVec3::ONE;
                let capacity = (dims.x as i64)
                    .saturating_mul(dims.y as i64)
                    .saturating_mul(dims.z as i64);
                if self.accelerated_contacts && count <= 8192 && (1..=1_000_000).contains(&capacity)
                {
                    let cell_index = |p: glam::IVec3| {
                        (p.z as usize * dims.y as usize + p.y as usize) * dims.x as usize
                            + p.x as usize
                    };
                    self.contact_lookup.resize(capacity as usize, usize::MAX);
                    self.contact_lookup.fill(usize::MAX);
                    self.contact_rows.resize(count, 0);
                    self.contact_occupied.clear();
                    for (i, &p) in self.contact_cells.iter().enumerate() {
                        let slot = cell_index(p - min_cell);
                        if self.contact_lookup[slot] == usize::MAX {
                            self.contact_lookup[slot] = self.contact_occupied.len();
                            self.contact_occupied.push(p - min_cell);
                        }
                        self.contact_rows[i] = self.contact_lookup[slot];
                    }
                    // Many gold grains share a riffle cell. Build its unchanged
                    // 27-cell candidate set once, rather than once per grain.
                    let words = count.div_ceil(64);
                    let rows = self.contact_occupied.len();
                    self.contact_members.resize(rows * words, 0);
                    self.contact_members.fill(0);
                    self.contact_candidates.resize(rows * words, 0);
                    self.contact_candidates.fill(0);
                    for i in 0..count {
                        self.contact_members[self.contact_rows[i] * words + i / 64] |=
                            1u64 << (i % 64);
                    }
                    for (row, &p) in self.contact_occupied.iter().enumerate() {
                        for x in -1..=1 {
                            for y in -1..=1 {
                                for z in -1..=1 {
                                    let other = self.contact_lookup
                                        [cell_index(p + glam::IVec3::new(x, y, z))];
                                    if other != usize::MAX {
                                        for word in 0..words {
                                            self.contact_candidates[row * words + word] |=
                                                self.contact_members[other * words + word];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    self.contact_lanes
                        .resize(count.div_ceil(4), [glam::Vec4::ZERO; 4]);
                    for i in 0..count {
                        self.update_contact_lane(i);
                    }
                    for i in 0..count {
                        let row = self.contact_rows[i];
                        for word in i / 64..words {
                            let mut value = self.contact_candidates[row * words + word];
                            if word == i / 64 {
                                value &= if i % 64 == 63 {
                                    0
                                } else {
                                    u64::MAX << ((i % 64) + 1)
                                };
                            }
                            while value != 0 {
                                let group = value.trailing_zeros() as usize / 4;
                                let block = word * 16 + group;
                                let mut mask = ((value >> (group * 4)) & 15) as u32;
                                value &= !(15u64 << (group * 4));
                                while mask != 0 {
                                    let c = &self.clumps[i];
                                    let radius = self.templates[c.template_idx].bounding_radius;
                                    let lane = self.contact_lanes[block];
                                    let dx = lane[0] - glam::Vec4::splat(c.position.x);
                                    let dy = lane[1] - glam::Vec4::splat(c.position.y);
                                    let dz = lane[2] - glam::Vec4::splat(c.position.z);
                                    let r = lane[3] + glam::Vec4::splat(radius);
                                    // Conservative rounding margin; the original scalar
                                    // narrow phase still makes the final rejection.
                                    let limit = r * r * glam::Vec4::splat(1. + 8. * f32::EPSILON)
                                        + glam::Vec4::splat(f32::MIN_POSITIVE);
                                    let nearby =
                                        ((dx * dx + dy * dy) + dz * dz).cmple(limit).bitmask()
                                            & mask;
                                    if nearby == 0 {
                                        break;
                                    }
                                    let lane = nearby.trailing_zeros() as usize;
                                    let j = block * 4 + lane;
                                    mask &= !((1u32 << (lane + 1)) - 1);
                                    self.resolve_dem_pair(i, j, percent);
                                    self.update_contact_lane(i);
                                    self.update_contact_lane(j);
                                }
                            }
                        }
                    }
                    continue;
                } else {
                    let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
                    for (i, c) in self.clumps.iter().enumerate() {
                        let p = (c.position / diameter).floor().as_ivec3();
                        cells.entry((p.x, p.y, p.z)).or_default().push(i);
                    }
                    for (i, c) in self.clumps.iter().enumerate() {
                        let p = (c.position / diameter).floor().as_ivec3();
                        for x in -1..=1 {
                            for y in -1..=1 {
                                for z in -1..=1 {
                                    if let Some(indices) = cells.get(&(p.x + x, p.y + y, p.z + z)) {
                                        for &j in indices {
                                            if j > i {
                                                pairs.push((i, j));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                pairs.sort_unstable();
            }
            for &(i, j) in &pairs {
                self.resolve_dem_pair(i, j, percent);
            }
        }
        self.contact_pairs = pairs;
    }

    fn update_contact_lane(&mut self, i: usize) {
        let c = &self.clumps[i];
        let r = self.templates[c.template_idx].bounding_radius;
        let lane = &mut self.contact_lanes[i / 4];
        lane[0][i % 4] = c.position.x;
        lane[1][i % 4] = c.position.y;
        lane[2][i % 4] = c.position.z;
        lane[3][i % 4] = r;
    }

    #[inline(always)]
    fn resolve_dem_pair(&mut self, i: usize, j: usize, percent: f32) {
        let (left, right) = self.clumps.split_at_mut(j);
        let clump_a = &mut left[i];
        let clump_b = &mut right[0];
        let template_a = &self.templates[clump_a.template_idx];
        let template_b = &self.templates[clump_b.template_idx];

        let delta = clump_b.position - clump_a.position;
        let max_dist = template_a.bounding_radius + template_b.bounding_radius;
        if delta.length_squared() > max_dist * max_dist {
            return;
        }

        let contact_dist = template_a.particle_radius + template_b.particle_radius;
        let contact_dist_sq = contact_dist * contact_dist;
        let slop = (contact_dist * 0.02).min(0.0005);
        let inv_mass_a = if self.accelerated_contacts {
            self.contact_inv_mass[clump_a.template_idx]
        } else {
            template_a.inv_mass()
        };
        let inv_mass_b = if self.accelerated_contacts {
            self.contact_inv_mass[clump_b.template_idx]
        } else {
            template_b.inv_mass()
        };
        let inv_mass_sum = inv_mass_a + inv_mass_b;
        if inv_mass_sum <= 0.0 {
            return;
        }

        // The playable DEM parcels are single centred spheres. Rotating a
        // zero offset and rebuilding the same centre distance for each contact
        // adds work without changing any collision, impulse or correction.
        if self.accelerated_contacts
            && template_a.local_offsets.as_slice() == [Vec3::ZERO]
            && template_b.local_offsets.as_slice() == [Vec3::ZERO]
        {
            let diff = delta;
            let dist_sq = diff.length_squared();

            if dist_sq >= contact_dist_sq {
                return;
            }

            let (normal, dist) = if dist_sq > 1.0e-10 {
                let dist = dist_sq.sqrt();
                (diff / dist, dist)
            } else {
                let center = clump_b.position - clump_a.position;
                let fallback = if center.length_squared() > 1.0e-10 {
                    center.normalize()
                } else {
                    Vec3::Y
                };
                (fallback, 0.0)
            };
            let penetration = contact_dist - dist;
            if penetration <= 0.0 {
                return;
            }
            let closing = (clump_b.velocity - clump_a.velocity).dot(normal);
            if self.coulomb_contacts {
                let contact =
                    (clump_a.position + normal * template_a.particle_radius + clump_b.position
                        - normal * template_b.particle_radius)
                        * 0.5;
                let point = crate::contact::ContactPoint {
                    normal,
                    arm_a: contact - clump_a.position,
                    arm_b: contact - clump_b.position,
                };
                let wet = self.contact_wet.get(i).copied().unwrap_or(false)
                    && self.contact_wet.get(j).copied().unwrap_or(false);
                crate::contact::resolve_pair(
                    clump_a,
                    template_a,
                    clump_b,
                    template_b,
                    point,
                    crate::contact::ContactMaterial {
                        friction: if wet {
                            self.wet_friction
                        } else {
                            self.friction
                        },
                        restitution: self.restitution,
                        rolling: 0.,
                    },
                );
            } else if closing < 0.0 {
                let impulse = normal * (-closing / inv_mass_sum);
                clump_a.velocity -= impulse * inv_mass_a;
                clump_b.velocity += impulse * inv_mass_b;
            }
            let correction_mag = (penetration - slop).max(0.0) * percent / inv_mass_sum;
            if correction_mag > 0.0 {
                let correction = normal * correction_mag;
                clump_a.position -= correction * inv_mass_a;
                clump_b.position += correction * inv_mass_b;
            }
            return;
        }
        for offset_a in &template_a.local_offsets {
            let ra = clump_a.rotation * *offset_a;
            let pa = clump_a.position + ra;
            for offset_b in &template_b.local_offsets {
                let rb = clump_b.rotation * *offset_b;
                let pb = clump_b.position + rb;
                let diff = pb - pa;
                let dist_sq = diff.length_squared();

                if dist_sq >= contact_dist_sq {
                    continue;
                }

                let (normal, dist) = if dist_sq > 1.0e-10 {
                    let dist = dist_sq.sqrt();
                    (diff / dist, dist)
                } else {
                    let center = clump_b.position - clump_a.position;
                    let fallback = if center.length_squared() > 1.0e-10 {
                        center.normalize()
                    } else {
                        Vec3::Y
                    };
                    (fallback, 0.0)
                };
                let penetration = contact_dist - dist;
                if penetration <= 0.0 {
                    continue;
                }
                let closing = (clump_b.velocity - clump_a.velocity).dot(normal);
                if self.coulomb_contacts {
                    let contact = (pa + normal * template_a.particle_radius + pb
                        - normal * template_b.particle_radius)
                        * 0.5;
                    let point = crate::contact::ContactPoint {
                        normal,
                        arm_a: contact - clump_a.position,
                        arm_b: contact - clump_b.position,
                    };
                    let wet = self.contact_wet.get(i).copied().unwrap_or(false)
                        && self.contact_wet.get(j).copied().unwrap_or(false);
                    crate::contact::resolve_pair(
                        clump_a,
                        template_a,
                        clump_b,
                        template_b,
                        point,
                        crate::contact::ContactMaterial {
                            friction: if wet {
                                self.wet_friction
                            } else {
                                self.friction
                            },
                            restitution: self.restitution,
                            rolling: 0.,
                        },
                    );
                } else if closing < 0.0 {
                    let impulse = normal * (-closing / inv_mass_sum);
                    clump_a.velocity -= impulse * inv_mass_a;
                    clump_b.velocity += impulse * inv_mass_b;
                }
                let correction_mag = (penetration - slop).max(0.0) * percent / inv_mass_sum;
                if correction_mag > 0.0 {
                    let correction = normal * correction_mag;
                    clump_a.position -= correction * inv_mass_a;
                    clump_b.position += correction * inv_mass_b;
                }
            }
        }
    }

    fn resolve_bounds(&mut self, dt: f32) {
        for clump in &mut self.clumps {
            let template = &self.templates[clump.template_idx];
            let mut min = Vec3::splat(f32::MAX);
            let mut max = Vec3::splat(f32::MIN);
            for offset in &template.local_offsets {
                let pos = clump.particle_world_position(*offset);
                min = min.min(pos);
                max = max.max(pos);
            }
            let r = Vec3::splat(template.particle_radius);
            min -= r;
            max += r;

            if min.x < self.bounds_min.x {
                clump.position.x += self.bounds_min.x - min.x;
                if clump.velocity.x < 0.0 {
                    clump.velocity.x = -clump.velocity.x * self.restitution;
                }
            } else if max.x > self.bounds_max.x {
                clump.position.x += self.bounds_max.x - max.x;
                if clump.velocity.x > 0.0 {
                    clump.velocity.x = -clump.velocity.x * self.restitution;
                }
            }

            if min.z < self.bounds_min.z {
                clump.position.z += self.bounds_min.z - min.z;
                if clump.velocity.z < 0.0 {
                    clump.velocity.z = -clump.velocity.z * self.restitution;
                }
            } else if max.z > self.bounds_max.z {
                clump.position.z += self.bounds_max.z - max.z;
                if clump.velocity.z > 0.0 {
                    clump.velocity.z = -clump.velocity.z * self.restitution;
                }
            }

            if min.y < self.bounds_min.y {
                clump.position.y += self.bounds_min.y - min.y;
                if clump.velocity.y < 0.0 {
                    clump.velocity.y = -clump.velocity.y * self.restitution;
                    // Use exponential decay for frame-rate independence
                    let friction_damp = (-self.friction * 10.0 * dt).exp();
                    clump.velocity.x *= friction_damp;
                    clump.velocity.z *= friction_damp;
                    // Angular damping: rate ~21 preserves original 0.7 at 60Hz
                    clump.angular_velocity *= (-21.0 * dt).exp();
                }
            } else if max.y > self.bounds_max.y {
                clump.position.y += self.bounds_max.y - max.y;
                if clump.velocity.y > 0.0 {
                    clump.velocity.y = -clump.velocity.y * self.restitution;
                }
            }
        }
    }

    fn resolve_clump_contacts(&mut self) {
        let count = self.clumps.len();
        let restitution = self.restitution;
        for i in 0..count {
            for j in (i + 1)..count {
                let (left, right) = self.clumps.split_at_mut(j);
                let clump_a = &mut left[i];
                let clump_b = &mut right[0];
                let template_a = &self.templates[clump_a.template_idx];
                let template_b = &self.templates[clump_b.template_idx];

                let delta = clump_b.position - clump_a.position;
                let max_dist = template_a.bounding_radius + template_b.bounding_radius;
                if delta.length_squared() > max_dist * max_dist {
                    continue;
                }

                Self::resolve_pair_contacts(restitution, clump_a, template_a, clump_b, template_b);
            }
        }
    }

    fn resolve_pair_contacts(
        restitution: f32,
        clump_a: &mut Clump3D,
        template_a: &ClumpTemplate3D,
        clump_b: &mut Clump3D,
        template_b: &ClumpTemplate3D,
    ) {
        let inv_mass_a = template_a.inv_mass();
        let inv_mass_b = template_b.inv_mass();
        if inv_mass_a == 0.0 && inv_mass_b == 0.0 {
            return;
        }

        let inv_inertia_a = clump_a.world_inertia_inv(template_a);
        let inv_inertia_b = clump_b.world_inertia_inv(template_b);

        let contact_dist = template_a.particle_radius + template_b.particle_radius;
        let contact_dist_sq = contact_dist * contact_dist;

        for offset_a in &template_a.local_offsets {
            let ra = clump_a.rotation * *offset_a;
            let pa = clump_a.position + ra;
            for offset_b in &template_b.local_offsets {
                let rb = clump_b.rotation * *offset_b;
                let pb = clump_b.position + rb;
                let diff = pb - pa;
                let dist_sq = diff.length_squared();

                if dist_sq < contact_dist_sq && dist_sq > 1.0e-10 {
                    let dist = dist_sq.sqrt();
                    let normal = diff / dist;
                    let penetration = contact_dist - dist;

                    let va = clump_a.particle_world_velocity(ra);
                    let vb = clump_b.particle_world_velocity(rb);
                    let rel_vel = vb - va;
                    let rel_normal = rel_vel.dot(normal);
                    if rel_normal > 0.0 {
                        continue;
                    }

                    let ra_cross_n = ra.cross(normal);
                    let rb_cross_n = rb.cross(normal);
                    let ang_a = (inv_inertia_a * ra_cross_n).cross(ra);
                    let ang_b = (inv_inertia_b * rb_cross_n).cross(rb);
                    let denom = inv_mass_a + inv_mass_b + normal.dot(ang_a + ang_b);
                    if denom <= 1.0e-6 {
                        continue;
                    }

                    let impulse_mag = -(1.0 + restitution) * rel_normal / denom;
                    let impulse = normal * impulse_mag;

                    clump_a.velocity -= impulse * inv_mass_a;
                    clump_b.velocity += impulse * inv_mass_b;
                    clump_a.angular_velocity -= inv_inertia_a * ra.cross(impulse);
                    clump_b.angular_velocity += inv_inertia_b * rb.cross(impulse);

                    let slop = 0.001;
                    let correction_mag =
                        (penetration - slop).max(0.0) * 0.4 / (inv_mass_a + inv_mass_b);
                    let correction = normal * correction_mag;
                    clump_a.position -= correction * inv_mass_a;
                    clump_b.position += correction * inv_mass_b;
                }
            }
        }
    }

    /// Apply fluid forces to DEM particles using Shields stress criterion.
    ///
    /// This method calculates the Shields parameter for each particle based on
    /// local fluid velocity, and applies entrainment forces only when the
    /// Shields stress exceeds the critical threshold.
    ///
    /// # Physics
    ///
    /// The Shields parameter θ* compares bed shear stress to particle weight:
    /// ```text
    /// θ* = τ / ((ρs - ρf) × g × d)
    /// ```
    ///
    /// Where:
    /// - τ = bed shear stress = Cf × ρf × U² (from friction coefficient)
    /// - ρs = sediment density (from particle mass/volume)
    /// - ρf = water density
    /// - g = gravity magnitude
    /// - d = grain diameter (2 × particle radius)
    ///
    /// When θ* > θ_cr (critical Shields):
    /// - Drag force: F_d = 0.5 × Cd × ρf × A × |U_rel|² × dir(U_rel)
    /// - Lift force: F_l = 0.5 × Cl × ρf × A × U² (upward)
    ///
    /// When θ* < θ_cr:
    /// - Particle remains stationary (no entrainment force)
    ///
    /// Buoyancy is always applied: F_b = ρf × V × g (upward)
    ///
    /// # Arguments
    ///
    /// * `dt` - Time step
    /// * `fluid_field` - Fluid velocity field for sampling
    /// * `fluid_params` - Physical parameters for fluid coupling
    ///
    /// # Returns
    ///
    /// A vector of (Shields parameter, was_entrained) tuples for each clump,
    /// useful for debugging and visualization.
    pub fn apply_fluid_forces_with_shields(
        &mut self,
        dt: f32,
        fluid_field: &FluidVelocityField,
        fluid_params: &FluidParams,
    ) -> Vec<(f32, bool)> {
        let g = self.gravity.length();
        let rho_f = fluid_params.water_density;
        let cf = fluid_params.bed_friction_coefficient;
        let theta_cr = fluid_params.critical_shields;
        let cd = fluid_params.drag_coefficient;
        let cl = fluid_params.lift_coefficient;

        let mut results = Vec::with_capacity(self.clumps.len());

        for clump in &mut self.clumps {
            let template = &self.templates[clump.template_idx];

            // Particle properties
            let radius = template.particle_radius;
            let diameter = radius * 2.0;
            let volume = (4.0 / 3.0) * PI * radius.powi(3) * template.local_offsets.len() as f32;
            let area = PI * template.bounding_radius.powi(2);
            let mass = template.mass;

            // Estimate particle density from mass and volume
            let rho_s = mass / volume;
            let rho_diff = rho_s - rho_f;

            // Sample fluid velocity at particle position
            let fluid_vel = sample_fluid_velocity(fluid_field, clump.position);
            let fluid_speed = fluid_vel.length();

            // Calculate bed shear stress: τ = Cf × ρf × U²
            let tau = cf * rho_f * fluid_speed * fluid_speed;

            // Calculate Shields parameter: θ* = τ / ((ρs - ρf) × g × d)
            // Avoid division by zero for very fine particles or neutral buoyancy
            let theta = if rho_diff > 0.0 && diameter > 1e-6 {
                tau / (rho_diff * g * diameter)
            } else {
                0.0
            };

            // Always apply buoyancy: F_b = ρf × V × g (upward)
            let buoyancy_force = Vec3::new(0.0, rho_f * volume * g, 0.0);
            clump.velocity += buoyancy_force * dt / mass;

            // Check if Shields stress exceeds critical value
            let entrained = theta > theta_cr;

            if entrained && fluid_speed > 1e-4 {
                // Particle is being entrained - apply drag and lift forces
                let relative_vel = fluid_vel - clump.velocity;
                let relative_speed = relative_vel.length();

                if relative_speed > 1e-4 {
                    // Drag force: F_d = 0.5 × Cd × ρf × A × |U_rel|²
                    // Scales with (θ - θ_cr) / θ_cr for gradual entrainment
                    let excess_shields = (theta - theta_cr) / theta_cr.max(0.001);
                    let entrainment_factor = excess_shields.min(1.0);

                    let drag_mag = 0.5 * cd * rho_f * area * relative_speed * relative_speed;
                    let drag_force = relative_vel.normalize() * drag_mag * entrainment_factor;
                    clump.velocity += drag_force * dt / mass;

                    // Lift force: F_l = 0.5 × Cl × ρf × A × U² (upward only)
                    // Only significant for particles near the bed
                    let lift_mag = 0.5 * cl * rho_f * area * fluid_speed * fluid_speed;
                    let lift_force = Vec3::new(0.0, lift_mag * entrainment_factor, 0.0);
                    clump.velocity += lift_force * dt / mass;
                }
            }

            results.push((theta, entrained));
        }

        results
    }

    /// Calculate Shields parameter for a single clump at its current position.
    ///
    /// Useful for diagnostics and visualization without applying forces.
    pub fn shields_parameter(
        &self,
        clump_idx: usize,
        fluid_field: &FluidVelocityField,
        fluid_params: &FluidParams,
    ) -> f32 {
        let clump = &self.clumps[clump_idx];
        let template = &self.templates[clump.template_idx];

        let g = self.gravity.length();
        let rho_f = fluid_params.water_density;
        let cf = fluid_params.bed_friction_coefficient;

        let radius = template.particle_radius;
        let diameter = radius * 2.0;
        let volume = (4.0 / 3.0) * PI * radius.powi(3) * template.local_offsets.len() as f32;
        let rho_s = template.mass / volume;
        let rho_diff = rho_s - rho_f;

        let fluid_vel = sample_fluid_velocity(fluid_field, clump.position);
        let fluid_speed = fluid_vel.length();
        let tau = cf * rho_f * fluid_speed * fluid_speed;

        if rho_diff > 0.0 && diameter > 1e-6 {
            tau / (rho_diff * g * diameter)
        } else {
            0.0
        }
    }

    /// Check if a clump would be entrained at its current position.
    pub fn is_entrained(
        &self,
        clump_idx: usize,
        fluid_field: &FluidVelocityField,
        fluid_params: &FluidParams,
    ) -> bool {
        let theta = self.shields_parameter(clump_idx, fluid_field, fluid_params);
        theta > fluid_params.critical_shields
    }
}

/// Sample fluid velocity from a MAC grid at a given world position.
fn sample_fluid_velocity(field: &FluidVelocityField, pos: Vec3) -> Vec3 {
    let local_pos = pos - field.grid_offset;
    let cell_size = field.cell_size;
    let width = field.grid_width;
    let height = field.grid_height;
    let depth = field.grid_depth;

    // Trilinear interpolation for each velocity component

    // U component (staggered on X faces)
    let u_pos = local_pos / cell_size - Vec3::new(0.0, 0.5, 0.5);
    let u = trilinear_sample(field.grid_u, u_pos, width + 1, height, depth);

    // V component (staggered on Y faces)
    let v_pos = local_pos / cell_size - Vec3::new(0.5, 0.0, 0.5);
    let v = trilinear_sample(field.grid_v, v_pos, width, height + 1, depth);

    // W component (staggered on Z faces)
    let w_pos = local_pos / cell_size - Vec3::new(0.5, 0.5, 0.0);
    let w = trilinear_sample(field.grid_w, w_pos, width, height, depth + 1);

    Vec3::new(u, v, w)
}

/// Trilinear interpolation for a scalar field.
fn trilinear_sample(field: &[f32], pos: Vec3, width: usize, height: usize, depth: usize) -> f32 {
    let x = pos.x.max(0.0);
    let y = pos.y.max(0.0);
    let z = pos.z.max(0.0);

    let i0 = (x.floor() as usize).min(width.saturating_sub(2));
    let j0 = (y.floor() as usize).min(height.saturating_sub(2));
    let k0 = (z.floor() as usize).min(depth.saturating_sub(2));

    let i1 = (i0 + 1).min(width - 1);
    let j1 = (j0 + 1).min(height - 1);
    let k1 = (k0 + 1).min(depth - 1);

    let fx = (x - i0 as f32).clamp(0.0, 1.0);
    let fy = (y - j0 as f32).clamp(0.0, 1.0);
    let fz = (z - k0 as f32).clamp(0.0, 1.0);

    let idx = |i: usize, j: usize, k: usize| -> usize { k * width * height + j * width + i };

    // Bounds check
    if idx(i1, j1, k1) >= field.len() {
        return 0.0;
    }

    let c000 = field[idx(i0, j0, k0)];
    let c100 = field[idx(i1, j0, k0)];
    let c010 = field[idx(i0, j1, k0)];
    let c110 = field[idx(i1, j1, k0)];
    let c001 = field[idx(i0, j0, k1)];
    let c101 = field[idx(i1, j0, k1)];
    let c011 = field[idx(i0, j1, k1)];
    let c111 = field[idx(i1, j1, k1)];

    let c00 = c000 * (1.0 - fx) + c100 * fx;
    let c10 = c010 * (1.0 - fx) + c110 * fx;
    let c01 = c001 * (1.0 - fx) + c101 * fx;
    let c11 = c011 * (1.0 - fx) + c111 * fx;

    let c0 = c00 * (1.0 - fy) + c10 * fy;
    let c1 = c01 * (1.0 - fy) + c11 * fy;

    c0 * (1.0 - fz) + c1 * fz
}

fn dem_damping(restitution: f32, stiffness: f32, mass: f32) -> f32 {
    if stiffness <= 0.0 || mass <= 0.0 {
        return 0.0;
    }
    let e = restitution.clamp(0.0, 0.999);
    if e <= 0.0 {
        return 2.0 * (stiffness * mass).sqrt();
    }
    let ln_e = e.ln();
    let zeta = -ln_e / (PI * PI + ln_e * ln_e).sqrt();
    2.0 * zeta * (stiffness * mass).sqrt()
}

fn center_on_com(offsets: &mut [Vec3]) {
    if offsets.is_empty() {
        return;
    }
    let sum = offsets.iter().fold(Vec3::ZERO, |acc, v| acc + *v);
    let com = sum / offsets.len() as f32;
    for offset in offsets {
        *offset -= com;
    }
}

fn compute_inertia(offsets: &[Vec3], particle_mass: f32, particle_radius: f32) -> (Mat3, f32) {
    let mut inertia = Mat3::ZERO;
    for r in offsets {
        let r2 = r.length_squared();
        let diag = Mat3::from_diagonal(Vec3::splat(r2));
        let outer = Mat3::from_cols(*r * r.x, *r * r.y, *r * r.z);
        inertia += (diag - outer) * particle_mass;
    }

    let jitter = (particle_mass * particle_radius * particle_radius).max(1.0e-6);
    inertia += Mat3::from_diagonal(Vec3::splat(jitter));

    let det = inertia.determinant();
    let inertia_inv = if det.abs() > 1.0e-6 {
        inertia.inverse()
    } else {
        let mass = particle_mass * offsets.len() as f32;
        let denom = (0.4 * mass * particle_radius * particle_radius).max(1.0e-6);
        Mat3::from_diagonal(Vec3::splat(1.0 / denom))
    };

    let bounding_radius =
        offsets.iter().map(|r| r.length()).fold(0.0_f32, f32::max) + particle_radius;

    (inertia_inv, bounding_radius)
}

fn generate_irregular_offsets(
    count: usize,
    spacing: f32,
    seed: u64,
    style: IrregularStyle3D,
) -> Vec<Vec3> {
    let mut rng = StdRng::seed_from_u64(seed);
    let spread = spacing * (count as f32).powf(1.0 / 3.0);
    let mut offsets = Vec::with_capacity(count);
    for _ in 0..count {
        let local = match style {
            IrregularStyle3D::Round => random_in_sphere(&mut rng),
            IrregularStyle3D::Sharp => random_sharp(&mut rng),
        };
        let offset = local * spread;
        offsets.push(offset);
    }
    offsets
}

fn random_in_sphere(rng: &mut StdRng) -> Vec3 {
    loop {
        let v = Vec3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        if v.length_squared() <= 1.0 {
            return v;
        }
    }
}

fn random_sharp(rng: &mut StdRng) -> Vec3 {
    let mut v = Vec3::new(
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    );
    if v.length_squared() < 1.0e-6 {
        v = Vec3::new(1.0, 0.0, 0.0);
    }

    if rng.gen_bool(0.6) {
        v.x = v.x.signum() * rng.gen_range(0.5..1.0);
        v.y = v.y.signum() * rng.gen_range(0.5..1.0);
        v.z = v.z.signum() * rng.gen_range(0.5..1.0);
    }

    let axis = rng.gen_range(0..3);
    let spike = rng.gen_range(1.2..1.8);
    match axis {
        0 => v.x *= spike,
        1 => v.y *= spike,
        _ => v.z *= spike,
    }

    let max_len = 1.6;
    let len = v.length();
    if len > max_len {
        v *= max_len / len;
    }

    v
}

/// Sample SDF at a world position and compute gradient (normal pointing away from solid)
///
/// # Arguments
/// * `sdf` - The signed distance field array
/// * `world_pos` - Position in world coordinates
/// * `grid_offset` - World position of grid origin (subtracted from world_pos to get grid-local pos)
/// * `width/height/depth` - Grid dimensions in cells
/// * `cell_size` - Size of each grid cell in world units
pub fn sample_sdf_with_gradient(
    sdf: &[f32],
    world_pos: Vec3,
    grid_offset: Vec3,
    width: usize,
    height: usize,
    depth: usize,
    cell_size: f32,
) -> (f32, Vec3) {
    sample_sdf_contact(
        &SdfParams {
            sdf,
            grid_offset,
            grid_width: width,
            grid_height: height,
            grid_depth: depth,
            cell_size,
        },
        world_pos,
        f32::INFINITY,
    )
}
fn sample_sdf_contact(params: &SdfParams, world_pos: Vec3, radius: f32) -> (f32, Vec3) {
    let SdfParams {
        sdf,
        grid_offset,
        grid_width: width,
        grid_height: height,
        grid_depth: depth,
        cell_size,
    } = *params;
    // Convert world position to grid-local position
    let local_pos = world_pos - grid_offset;

    // Convert grid-local position to grid coordinates (cell indices)
    // SDF samples are at CELL CENTRES, exactly as Grid3D and the GPU sampler.
    let fx = local_pos.x / cell_size - 0.5;
    let fy = local_pos.y / cell_size - 0.5;
    let fz = local_pos.z / cell_size - 0.5;

    // Clamp to valid range
    let fx = fx.clamp(0.0, width as f32 - 1.0);
    let fy = fy.clamp(0.0, height as f32 - 1.0);
    let fz = fz.clamp(0.0, depth as f32 - 1.0);

    let i = fx as usize;
    let j = fy as usize;
    let k = fz as usize;

    // Trilinear interpolation weights
    let tx = fx - i as f32;
    let ty = fy - j as f32;
    let tz = fz - k as f32;

    // Sample 8 corners of the cell
    let idx = |ii: usize, jj: usize, kk: usize| -> f32 {
        let ii = ii.min(width - 1);
        let jj = jj.min(height - 1);
        let kk = kk.min(depth - 1);
        let index = kk * width * height + jj * width + ii;
        if index < sdf.len() {
            sdf[index]
        } else {
            1.0 // Outside bounds = far from solid
        }
    };

    let c000 = idx(i, j, k);
    let c100 = idx(i + 1, j, k);
    let c010 = idx(i, j + 1, k);
    let c110 = idx(i + 1, j + 1, k);
    let c001 = idx(i, j, k + 1);
    let c101 = idx(i + 1, j, k + 1);
    let c011 = idx(i, j + 1, k + 1);
    let c111 = idx(i + 1, j + 1, k + 1);

    // Trilinear interpolation for SDF value
    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;

    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;

    let sdf_value = c0 * (1.0 - tz) + c1 * tz;

    if sdf_value >= radius {
        return (sdf_value, Vec3::ZERO);
    }

    // Differentiate the SAME trilinear scalar field. Nearest-node finite
    // differences produced discontinuous normals unrelated to the contact SDF.
    let grad_x = ((c100 - c000) * (1.0 - ty) * (1.0 - tz)
        + (c110 - c010) * ty * (1.0 - tz)
        + (c101 - c001) * (1.0 - ty) * tz
        + (c111 - c011) * ty * tz)
        / cell_size;
    let grad_y = ((c10 - c00) * (1.0 - tz) + (c11 - c01) * tz) / cell_size;
    let grad_z = (c1 - c0) / cell_size;

    (sdf_value, Vec3::new(grad_x, grad_y, grad_z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contact_sdf_and_normal_use_cell_centres_and_the_same_trilinear_field() {
        let n = 5;
        let dx = 0.1;
        // An oblique plane detects both half-cell shifts and normal errors.
        let normal = Vec3::new(1., 2., 3.).normalize();
        let mut sdf = vec![0.; n * n * n];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let p = (Vec3::new(x as f32, y as f32, z as f32) + Vec3::splat(0.5)) * dx;
                    sdf[z * n * n + y * n + x] = normal.dot(p) - 0.6;
                }
            }
        }
        let params = SdfParams {
            sdf: &sdf,
            grid_width: n,
            grid_height: n,
            grid_depth: n,
            cell_size: dx,
            grid_offset: Vec3::ZERO,
        };
        let p = Vec3::new(0.184, 0.233, 0.279);
        let (distance, gradient) = sample_sdf_contact(&params, p, 1.);
        assert!((distance - (normal.dot(p) - 0.6)).abs() < 1e-7);
        assert!((gradient - normal).length() < 2e-6);
    }

    #[test]
    fn contact_normal_is_the_derivative_of_the_interpolated_distance() {
        let n = 5;
        let dx = 0.1;
        let f = |p: Vec3| p.x * p.y + 2. * p.y * p.z + 3. * p.z * p.x - 1.;
        let mut sdf = vec![0.; n * n * n];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    sdf[z * n * n + y * n + x] =
                        f((Vec3::new(x as f32, y as f32, z as f32) + Vec3::splat(0.5)) * dx);
                }
            }
        }
        let params = SdfParams {
            sdf: &sdf,
            grid_width: n,
            grid_height: n,
            grid_depth: n,
            cell_size: dx,
            grid_offset: Vec3::ZERO,
        };
        let p = Vec3::new(0.184, 0.233, 0.279);
        let (distance, gradient) = sample_sdf_contact(&params, p, 1.);
        let expected = Vec3::new(p.y + 3. * p.z, p.x + 2. * p.z, 2. * p.y + 3. * p.x);
        assert!((distance - f(p)).abs() < 1e-7);
        assert!((gradient - expected).length() < 3e-6);
    }

    // ========== ClumpTemplate3D Tests ==========

    #[test]
    fn test_template_tetra_shape() {
        let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.05, 1.0);
        assert_eq!(template.local_offsets.len(), 4);
        assert!(template.bounding_radius > 0.0);
        assert_eq!(template.mass, 4.0); // 4 particles * 1.0 mass each
        assert_eq!(template.particle_radius, 0.05);
        assert_eq!(template.particle_mass, 1.0);
    }

    #[test]
    fn test_template_cube2_shape() {
        let template = ClumpTemplate3D::generate(ClumpShape3D::Cube2, 0.05, 1.0);
        assert_eq!(template.local_offsets.len(), 8);
        assert!(template.bounding_radius > 0.0);
        assert_eq!(template.mass, 8.0);
    }

    #[test]
    fn test_template_flat4_shape() {
        let template = ClumpTemplate3D::generate(ClumpShape3D::Flat4, 0.05, 1.0);
        assert_eq!(template.local_offsets.len(), 4);
        // Flat4 should have all particles at same Y level
        let y_values: Vec<f32> = template.local_offsets.iter().map(|o| o.y).collect();
        let y_spread = y_values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            - y_values
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
        assert!(y_spread.abs() < 1e-5, "Flat4 should be planar in XZ");
    }

    #[test]
    fn test_template_rod3_shape() {
        let template = ClumpTemplate3D::generate(ClumpShape3D::Rod3, 0.05, 1.0);
        assert_eq!(template.local_offsets.len(), 3);
        // Rod3 should be linear along X axis
        for offset in &template.local_offsets {
            assert!(offset.y.abs() < 1e-5);
            assert!(offset.z.abs() < 1e-5);
        }
    }

    #[test]
    fn test_template_irregular_round() {
        let template = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 5,
                seed: 42,
                style: IrregularStyle3D::Round,
            },
            0.05,
            1.0,
        );
        assert_eq!(template.local_offsets.len(), 5);
        assert_eq!(template.mass, 5.0);
    }

    #[test]
    fn test_template_irregular_sharp() {
        let template = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 7,
                seed: 123,
                style: IrregularStyle3D::Sharp,
            },
            0.05,
            1.0,
        );
        assert_eq!(template.local_offsets.len(), 7);
        assert_eq!(template.mass, 7.0);
    }

    #[test]
    fn test_template_irregular_deterministic() {
        // Same seed should produce same offsets
        let t1 = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 4,
                seed: 999,
                style: IrregularStyle3D::Round,
            },
            0.05,
            1.0,
        );
        let t2 = ClumpTemplate3D::generate(
            ClumpShape3D::Irregular {
                count: 4,
                seed: 999,
                style: IrregularStyle3D::Round,
            },
            0.05,
            1.0,
        );
        for (o1, o2) in t1.local_offsets.iter().zip(t2.local_offsets.iter()) {
            assert!((*o1 - *o2).length() < 1e-6);
        }
    }

    #[test]
    fn test_template_centered_on_com() {
        let template = ClumpTemplate3D::generate(ClumpShape3D::Cube2, 0.05, 1.0);
        let com: Vec3 = template
            .local_offsets
            .iter()
            .fold(Vec3::ZERO, |acc, v| acc + *v)
            / template.local_offsets.len() as f32;
        assert!(com.length() < 1e-5, "Template should be centered on COM");
    }

    #[test]
    fn test_template_inv_mass() {
        let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.05, 2.0);
        assert_eq!(template.mass, 8.0); // 4 particles * 2.0
        assert!((template.inv_mass() - 1.0 / 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_template_inv_mass_zero() {
        let mut template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.05, 1.0);
        template.mass = 0.0;
        assert_eq!(template.inv_mass(), 0.0);
    }

    #[test]
    fn test_template_inertia_valid() {
        let template = ClumpTemplate3D::generate(ClumpShape3D::Cube2, 0.05, 1.0);
        // Inverse inertia should have finite, positive diagonal elements
        let diag = Vec3::new(
            template.inertia_inv_local.x_axis.x,
            template.inertia_inv_local.y_axis.y,
            template.inertia_inv_local.z_axis.z,
        );
        assert!(diag.x > 0.0 && diag.x.is_finite());
        assert!(diag.y > 0.0 && diag.y.is_finite());
        assert!(diag.z > 0.0 && diag.z.is_finite());
    }

    // ========== Clump3D Tests ==========

    #[test]
    fn test_clump_new() {
        let clump = Clump3D::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, 0.0, -0.5), 0);
        assert_eq!(clump.position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(clump.velocity, Vec3::new(0.5, 0.0, -0.5));
        assert_eq!(clump.rotation, Quat::IDENTITY);
        assert_eq!(clump.angular_velocity, Vec3::ZERO);
        assert_eq!(clump.template_idx, 0);
    }

    #[test]
    fn test_clump_world_position_no_rotation() {
        let clump = Clump3D::new(Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO, 0);
        let offset = Vec3::new(0.1, 0.2, 0.3);
        let world_pos = clump.particle_world_position(offset);
        assert!((world_pos - Vec3::new(1.1, 2.2, 3.3)).length() < 1e-6);
    }

    #[test]
    fn test_clump_world_position_with_rotation() {
        let mut clump = Clump3D::new(Vec3::ZERO, Vec3::ZERO, 0);
        // Rotate 90 degrees around Y axis
        clump.rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let offset = Vec3::new(1.0, 0.0, 0.0);
        let world_pos = clump.particle_world_position(offset);
        // After 90deg Y rotation, X becomes Z
        assert!((world_pos.x - 0.0).abs() < 1e-5);
        assert!((world_pos.z - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_clump_world_velocity_linear_only() {
        let clump = Clump3D::new(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0), 0);
        let vel = clump.particle_world_velocity(Vec3::new(0.1, 0.0, 0.0));
        // With zero angular velocity, world velocity equals linear velocity
        assert!((vel - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-6);
    }

    #[test]
    fn test_clump_world_velocity_with_angular() {
        let mut clump = Clump3D::new(Vec3::ZERO, Vec3::ZERO, 0);
        clump.angular_velocity = Vec3::new(0.0, 1.0, 0.0); // Rotating around Y
        let offset_world = Vec3::new(1.0, 0.0, 0.0);
        let vel = clump.particle_world_velocity(offset_world);
        // omega × r = (0, 1, 0) × (1, 0, 0) = (0, 0, -1)
        assert!((vel.z - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_clump_world_inertia_identity_rotation() {
        let template = ClumpTemplate3D::generate(ClumpShape3D::Cube2, 0.05, 1.0);
        let clump = Clump3D::new(Vec3::ZERO, Vec3::ZERO, 0);
        let world_inertia = clump.world_inertia_inv(&template);
        // With identity rotation, world inertia equals local inertia
        for i in 0..3 {
            for j in 0..3 {
                let local = template.inertia_inv_local.col(i)[j];
                let world = world_inertia.col(i)[j];
                assert!((local - world).abs() < 1e-6);
            }
        }
    }

    // ========== ClusterSimulation3D Basic Tests ==========

    #[test]
    fn test_simulation_new() {
        let sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
        assert_eq!(sim.bounds_min, Vec3::ZERO);
        assert_eq!(sim.bounds_max, Vec3::new(10.0, 10.0, 10.0));
        assert!(sim.clumps.is_empty());
        assert!(sim.templates.is_empty());
        assert_eq!(sim.gravity, Vec3::new(0.0, -9.81, 0.0));
    }

    #[test]
    fn test_simulation_add_template() {
        let mut sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(10.0));
        let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.05, 1.0);
        let idx = sim.add_template(template);
        assert_eq!(idx, 0);
        assert_eq!(sim.templates.len(), 1);

        let template2 = ClumpTemplate3D::generate(ClumpShape3D::Cube2, 0.05, 1.0);
        let idx2 = sim.add_template(template2);
        assert_eq!(idx2, 1);
        assert_eq!(sim.templates.len(), 2);
    }

    #[test]
    fn test_simulation_spawn() {
        let mut sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(10.0));
        let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.05, 1.0);
        sim.add_template(template);

        let idx = sim.spawn(0, Vec3::new(5.0, 5.0, 5.0), Vec3::ZERO);
        assert_eq!(idx, 0);
        assert_eq!(sim.clumps.len(), 1);
        assert_eq!(sim.clumps[0].position, Vec3::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn test_simulation_contact_counts_initial() {
        let sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(10.0));
        assert_eq!(sim.sphere_contact_count(), 0);
        assert_eq!(sim.plane_contact_count(), 0);
    }

    // ========== Physics Simulation Tests ==========

    #[test]
    fn test_gravity_accelerates_downward() {
        let mut sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(100.0));
        let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.05, 1.0);
        sim.add_template(template);
        sim.spawn(0, Vec3::new(50.0, 50.0, 50.0), Vec3::ZERO);

        let initial_y = sim.clumps[0].position.y;
        sim.step(0.016); // ~60fps timestep

        // Should have moved downward due to gravity
        assert!(sim.clumps[0].position.y < initial_y);
        assert!(sim.clumps[0].velocity.y < 0.0);
    }

    #[test]
    fn test_floor_collision() {
        let mut sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(10.0));
        let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.05, 1.0);
        sim.add_template(template);

        // Spawn just above floor
        sim.spawn(0, Vec3::new(5.0, 0.5, 5.0), Vec3::new(0.0, -5.0, 0.0));

        // Run simulation for a while
        for _ in 0..100 {
            sim.step(0.016);
        }

        // Clump should not fall through floor
        let template = &sim.templates[0];
        let clump = &sim.clumps[0];
        for offset in &template.local_offsets {
            let world_pos = clump.particle_world_position(*offset);
            assert!(
                world_pos.y - template.particle_radius >= -0.01,
                "Particle should not penetrate floor"
            );
        }
    }

    #[test]
    fn test_wall_collision() {
        let mut sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(10.0));
        sim.gravity = Vec3::ZERO; // Disable gravity for this test
        let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.05, 1.0);
        sim.add_template(template);

        // Launch toward wall
        sim.spawn(0, Vec3::new(9.0, 5.0, 5.0), Vec3::new(10.0, 0.0, 0.0));

        for _ in 0..100 {
            sim.step(0.016);
        }

        // Should bounce off wall and stay in bounds
        assert!(sim.clumps[0].position.x <= 10.0);
    }

    #[test]
    fn test_energy_dissipation_with_restitution() {
        let mut sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(10.0));
        sim.restitution = 0.5;
        let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.05, 1.0);
        sim.add_template(template);

        // Drop from height
        sim.spawn(0, Vec3::new(5.0, 5.0, 5.0), Vec3::ZERO);

        // Run for a while
        for _ in 0..500 {
            sim.step(0.016);
        }

        // Should have settled (low velocity) due to energy loss
        let speed = sim.clumps[0].velocity.length();
        assert!(speed < 1.0, "Clump should settle due to energy dissipation");
    }

    // ========== Collision Detection Tests ==========

    #[test]
    fn test_sphere_sphere_collision_detection() {
        let mut sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(10.0));
        sim.gravity = Vec3::ZERO;
        let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.1, 1.0);
        sim.add_template(template);

        // Spawn two clumps moving toward each other
        sim.spawn(0, Vec3::new(4.0, 5.0, 5.0), Vec3::new(2.0, 0.0, 0.0));
        sim.spawn(0, Vec3::new(6.0, 5.0, 5.0), Vec3::new(-2.0, 0.0, 0.0));

        // Step until collision
        for _ in 0..50 {
            sim.step(0.016);
        }

        // Clumps should have bounced apart
        let dist = (sim.clumps[0].position - sim.clumps[1].position).length();
        assert!(dist > 0.1, "Clumps should have separated after collision");
    }

    #[test]
    fn test_has_contact() {
        let mut sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(10.0));
        sim.gravity = Vec3::ZERO;
        let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.2, 1.0);
        sim.add_template(template);

        // Spawn overlapping clumps
        sim.spawn(0, Vec3::new(5.0, 5.0, 5.0), Vec3::ZERO);
        sim.spawn(0, Vec3::new(5.3, 5.0, 5.0), Vec3::ZERO);

        sim.step(0.001); // Small step to detect contacts

        // Should detect contact between overlapping clumps
        // Note: has_contact checks sphere_contacts which are populated during DEM step
        let contact_count = sim.sphere_contact_count();
        assert!(contact_count > 0, "Should detect sphere-sphere contacts");
    }

    // ========== SDF Tests ==========

    #[test]
    fn test_sample_sdf_uniform_positive() {
        // SDF with uniform positive value (empty space)
        let sdf: Vec<f32> = vec![1.0; 8]; // 2x2x2 grid
        let (value, _gradient) =
            sample_sdf_with_gradient(&sdf, Vec3::new(0.5, 0.5, 0.5), Vec3::ZERO, 2, 2, 2, 1.0);
        assert!((value - 1.0).abs() < 0.1, "Uniform SDF should return ~1.0");
    }

    #[test]
    fn test_sample_sdf_floor_plane() {
        // Create a floor SDF: negative below y=2, positive above
        let width = 4;
        let height = 4;
        let depth = 4;
        let mut sdf = vec![0.0; width * height * depth];

        for k in 0..depth {
            for j in 0..height {
                for i in 0..width {
                    let idx = k * width * height + j * width + i;
                    sdf[idx] = j as f32 - 1.5; // Floor at y=1.5 cells
                }
            }
        }

        // Sample above floor
        let (value_above, grad_above) = sample_sdf_with_gradient(
            &sdf,
            Vec3::new(2.0, 3.0, 2.0),
            Vec3::ZERO,
            width,
            height,
            depth,
            1.0,
        );
        assert!(value_above > 0.0, "Above floor should be positive");
        assert!(
            grad_above.y > 0.0,
            "Gradient should point up (away from solid)"
        );

        // Sample below floor
        let (value_below, _) = sample_sdf_with_gradient(
            &sdf,
            Vec3::new(2.0, 0.5, 2.0),
            Vec3::ZERO,
            width,
            height,
            depth,
            1.0,
        );
        assert!(value_below < 0.0, "Below floor should be negative");
    }

    #[test]
    fn test_sample_sdf_with_grid_offset() {
        let sdf: Vec<f32> = vec![2.0; 8]; // 2x2x2 grid
        let grid_offset = Vec3::new(10.0, 20.0, 30.0);

        // Sample at world position that maps to grid center
        let (value, _) = sample_sdf_with_gradient(
            &sdf,
            Vec3::new(11.0, 21.0, 31.0), // grid_offset + (1,1,1)
            grid_offset,
            2,
            2,
            2,
            1.0,
        );
        assert!((value - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_step_with_sdf_floor() {
        let mut sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(10.0));
        let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.1, 1.0);
        sim.add_template(template);
        sim.spawn(0, Vec3::new(5.0, 2.0, 5.0), Vec3::ZERO);

        // Create floor SDF at y=1
        let width = 12;
        let height = 12;
        let depth = 12;
        let mut sdf = vec![10.0; width * height * depth];

        for k in 0..depth {
            for j in 0..height {
                for i in 0..width {
                    let idx = k * width * height + j * width + i;
                    let y = j as f32;
                    sdf[idx] = y - 1.0; // Floor at y=1 cell (y=1.0 world)
                }
            }
        }

        let sdf_params = SdfParams {
            sdf: &sdf,
            grid_width: width,
            grid_height: height,
            grid_depth: depth,
            cell_size: 1.0,
            grid_offset: Vec3::ZERO,
        };

        for _ in 0..200 {
            sim.step_with_sdf(0.016, &sdf_params);
        }

        // Clump should rest on the floor
        assert!(
            sim.clumps[0].position.y > 0.5,
            "Clump should not fall through SDF floor"
        );
    }

    // ========== Helper Function Tests ==========

    #[test]
    fn test_dem_damping_zero_restitution() {
        // Zero restitution = critical damping
        let damping = dem_damping(0.0, 1000.0, 1.0);
        let critical = 2.0 * (1000.0_f32 * 1.0).sqrt();
        assert!((damping - critical).abs() < 1e-3);
    }

    #[test]
    fn test_dem_damping_high_restitution() {
        // High restitution = low damping
        let damping_high = dem_damping(0.9, 1000.0, 1.0);
        let damping_low = dem_damping(0.1, 1000.0, 1.0);
        assert!(damping_high < damping_low);
    }

    #[test]
    fn test_dem_damping_invalid_inputs() {
        assert_eq!(dem_damping(0.5, 0.0, 1.0), 0.0);
        assert_eq!(dem_damping(0.5, 1000.0, 0.0), 0.0);
        assert_eq!(dem_damping(0.5, -1.0, 1.0), 0.0);
    }

    #[test]
    fn test_center_on_com_empty() {
        let mut offsets: Vec<Vec3> = vec![];
        center_on_com(&mut offsets);
        assert!(offsets.is_empty());
    }

    #[test]
    fn test_center_on_com_single() {
        let mut offsets = vec![Vec3::new(1.0, 2.0, 3.0)];
        center_on_com(&mut offsets);
        assert!(
            offsets[0].length() < 1e-6,
            "Single point should be at origin"
        );
    }

    #[test]
    fn test_center_on_com_multiple() {
        let mut offsets = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
        ];
        center_on_com(&mut offsets);
        let com: Vec3 = offsets.iter().fold(Vec3::ZERO, |acc, v| acc + *v) / 4.0;
        assert!(
            com.length() < 1e-6,
            "COM should be at origin after centering"
        );
    }

    #[test]
    fn test_compute_inertia_positive_definite() {
        let offsets = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        ];
        let (inertia_inv, bounding_radius) = compute_inertia(&offsets, 1.0, 0.1);

        // Inverse inertia should be positive definite (positive diagonal)
        assert!(inertia_inv.x_axis.x > 0.0);
        assert!(inertia_inv.y_axis.y > 0.0);
        assert!(inertia_inv.z_axis.z > 0.0);

        // Bounding radius should encompass all particles
        assert!(bounding_radius >= 1.0 + 0.1);
    }

    #[test]
    fn test_compute_inertia_symmetry() {
        // Symmetric configuration should have equal moments about each axis
        let offsets = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let (inertia_inv, _) = compute_inertia(&offsets, 1.0, 0.1);

        // Should be approximately equal due to symmetry
        let ixx = inertia_inv.x_axis.x;
        let iyy = inertia_inv.y_axis.y;
        let izz = inertia_inv.z_axis.z;
        assert!((ixx - iyy).abs() < 1e-5);
        assert!((iyy - izz).abs() < 1e-5);
    }

    // ========== Integration Tests ==========

    #[test]
    fn test_multiple_clumps_settle() {
        let mut sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::new(2.0, 5.0, 2.0));
        let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.05, 0.5);
        sim.add_template(template);

        // Stack clumps vertically
        for i in 0..3 {
            sim.spawn(0, Vec3::new(1.0, 1.0 + i as f32 * 0.5, 1.0), Vec3::ZERO);
        }

        // Let them settle
        for _ in 0..500 {
            sim.step(0.016);
        }

        // All clumps should be above floor and have low velocity
        for clump in &sim.clumps {
            assert!(clump.position.y > 0.0);
            assert!(clump.velocity.length() < 2.0);
        }
    }

    #[test]
    fn test_rotation_integration() {
        let mut sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(10.0));
        sim.gravity = Vec3::ZERO;
        sim.use_dem = false; // Disable DEM for pure rotation test

        let template = ClumpTemplate3D::generate(ClumpShape3D::Rod3, 0.05, 1.0);
        sim.add_template(template);
        sim.spawn(0, Vec3::new(5.0, 5.0, 5.0), Vec3::ZERO);

        // Set angular velocity
        sim.clumps[0].angular_velocity = Vec3::new(0.0, 1.0, 0.0);

        let initial_rotation = sim.clumps[0].rotation;
        sim.step(0.1);

        // Rotation should have changed
        let angle = initial_rotation.angle_between(sim.clumps[0].rotation);
        assert!(angle > 0.05, "Clump should have rotated");
    }

    #[test]
    fn test_collision_response_only_preserves_position() {
        let mut sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(10.0));
        let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.1, 1.0);
        sim.add_template(template);
        sim.spawn(0, Vec3::new(5.0, 5.0, 5.0), Vec3::ZERO);

        // Create empty SDF (no collision)
        let sdf: Vec<f32> = vec![10.0; 8];
        let sdf_params = SdfParams {
            sdf: &sdf,
            grid_width: 2,
            grid_height: 2,
            grid_depth: 2,
            cell_size: 5.0,
            grid_offset: Vec3::ZERO,
        };

        let pos_before = sim.clumps[0].position;
        sim.collision_response_only(0.016, &sdf_params, false);
        let pos_after = sim.clumps[0].position;

        // Without penetration, position should be unchanged
        assert!((pos_before - pos_after).length() < 0.01);
    }

    #[test]
    fn coupled_contacts_preserve_momentum_with_spatial_neighbours() {
        let run = |count: usize| {
            let mut sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(100.0));
            let a = sim.add_template(ClumpTemplate3D::generate(
                ClumpShape3D::Irregular {
                    count: 1,
                    seed: 0,
                    style: super::IrregularStyle3D::Round,
                },
                0.001,
                0.002,
            ));
            let b = sim.add_template(ClumpTemplate3D::generate(
                ClumpShape3D::Irregular {
                    count: 1,
                    seed: 0,
                    style: super::IrregularStyle3D::Round,
                },
                0.001,
                0.004,
            ));
            sim.spawn(a, Vec3::new(1.0, 1.0, 1.0), Vec3::X);
            sim.spawn(b, Vec3::new(1.0015, 1.0, 1.0), -Vec3::X);
            for i in 2..count {
                sim.spawn(a, Vec3::new(10.0 + i as f32 * 0.1, 1.0, 1.0), Vec3::ZERO);
            }
            sim.resolve_dem_penetrations(2);
            let momentum = sim.clumps[0].velocity * 0.002 + sim.clumps[1].velocity * 0.004;
            assert!((momentum - Vec3::new(-0.002, 0.0, 0.0)).length() < 1e-8);
            assert!((sim.clumps[1].velocity - sim.clumps[0].velocity).x >= -1e-6);
            assert!((sim.clumps[1].position - sim.clumps[0].position).length() > 0.0019);
            [
                sim.clumps[0].position,
                sim.clumps[1].position,
                sim.clumps[0].velocity,
                sim.clumps[1].velocity,
            ]
        };
        assert_eq!(run(2), run(65));
    }

    #[test]
    fn accelerated_search_matches_hash_contacts_exactly() {
        let run = |dense| {
            let mut sim = ClusterSimulation3D::new(Vec3::splat(-1.0), Vec3::splat(2.0));
            sim.accelerated_contacts = dense;
            for radius in [0.004, 0.007, 0.01] {
                sim.add_template(ClumpTemplate3D::generate(
                    ClumpShape3D::Tetra,
                    radius,
                    radius * 2.0,
                ));
            }
            for i in 0..180 {
                let f = |a: u32| ((i as u32 * 1664525 + a * 1013904223) % 1009) as f32 / 1009.0;
                sim.spawn(
                    i % 3,
                    Vec3::new(f(1), f(2), f(3)) * 0.12 - Vec3::splat(0.06),
                    Vec3::new(f(0), f(1), f(2)) - Vec3::splat(0.5),
                );
            }
            for _ in 0..3 {
                sim.resolve_dem_penetrations(2);
            }
            sim.clumps
                .into_iter()
                .map(|c| (c.position, c.velocity))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(true), run(false));
    }

    #[test]
    fn vector_contacts_match_reference_in_dense_mixed_size_pockets() {
        let run = |accelerated, seed: u32| {
            let mut sim = ClusterSimulation3D::new(Vec3::splat(-1.), Vec3::ONE);
            sim.accelerated_contacts = accelerated;
            for (shape, r) in [
                (ClumpShape3D::Tetra, 0.003),
                (ClumpShape3D::Tetra, 0.0001),
                (ClumpShape3D::Cube2, 0.002),
            ] {
                sim.add_template(ClumpTemplate3D::generate(shape, r, r * 3.));
            }
            for i in 0u32..600 {
                let f = |n: u32| {
                    i.wrapping_mul(1664525u32)
                        .wrapping_add(n.wrapping_mul(1013904223))
                        .wrapping_add(seed)
                        % 1009
                };
                let p = Vec3::new(f(0) as f32, f(1) as f32, f(2) as f32) * 0.000025;
                sim.spawn(
                    i as usize % 3,
                    p,
                    Vec3::new(f(3) as f32 * 0.0001, -0.1, 0.02),
                );
            }
            for _ in 0..4 {
                sim.resolve_dem_penetrations(2);
            }
            sim.clumps
                .into_iter()
                .map(|c| (c.position, c.velocity))
                .collect::<Vec<_>>()
        };
        for seed in [0, 17, 123] {
            assert_eq!(run(true, seed), run(false, seed));
        }
    }

    #[test]
    fn centred_sphere_fast_path_matches_rotated_reference_exactly() {
        let run = |accelerated| {
            let mut sim = ClusterSimulation3D::new(Vec3::splat(-1.), Vec3::ONE);
            sim.accelerated_contacts = accelerated;
            for radius in [0.00004, 0.0003, 0.006] {
                sim.add_template(ClumpTemplate3D::generate(
                    ClumpShape3D::Irregular {
                        count: 1,
                        seed: 0,
                        style: IrregularStyle3D::Round,
                    },
                    radius,
                    radius * 2.,
                ));
            }
            for i in 0u32..700 {
                let f = |n: u32| {
                    i.wrapping_mul(1664525u32)
                        .wrapping_add(n.wrapping_mul(1013904223))
                        % 1009
                };
                let index = sim.spawn(
                    i as usize % 3,
                    Vec3::new(f(0) as f32, f(1) as f32, f(2) as f32) * 0.000025,
                    Vec3::new(0.1, -0.1, 0.02),
                );
                sim.clumps[index].rotation = Quat::from_rotation_y(f(3) as f32 * 0.01);
            }
            for _ in 0..4 {
                sim.resolve_dem_penetrations(2);
            }
            sim.clumps
                .into_iter()
                .map(|c| (c.position, c.velocity))
                .collect::<Vec<_>>()
        };
        assert_eq!(run(true), run(false));
    }

    #[test]
    fn test_wet_friction_lower_than_dry() {
        let sim = ClusterSimulation3D::new(Vec3::ZERO, Vec3::splat(10.0));
        assert!(sim.wet_friction < sim.floor_friction);
        assert!(sim.wet_rolling_friction < sim.rolling_friction);
    }
}
