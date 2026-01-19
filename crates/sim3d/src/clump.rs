//! 3D rigid clumps (multi-sphere clusters) for rock/gravel prototypes.

use crate::serde_utils::{
    deserialize_mat3, deserialize_quat, deserialize_vec3, serialize_mat3, serialize_quat,
    serialize_vec3,
};
use glam::{Mat3, Quat, Vec3};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{collections::HashMap, f32::consts::PI};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum IrregularStyle3D {
    Round,
    Sharp,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClumpTemplate3D {
    pub shape: ClumpShape3D,
    #[serde(
        serialize_with = "serialize_vec3_vec",
        deserialize_with = "deserialize_vec3_vec"
    )]
    pub local_offsets: Vec<Vec3>,
    pub particle_radius: f32,
    pub particle_mass: f32,
    pub mass: f32,
    #[serde(serialize_with = "serialize_mat3", deserialize_with = "deserialize_mat3")]
    pub inertia_inv_local: Mat3,
    pub bounding_radius: f32,
}

// Helper for Vec<Vec3>
fn serialize_vec3_vec<S>(v: &Vec<Vec3>, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    use crate::serde_utils::Vec3Def;
    let defs: Vec<Vec3Def> = v.iter().map(|&vec| vec.into()).collect();
    defs.serialize(s)
}

fn deserialize_vec3_vec<'de, D>(d: D) -> Result<Vec<Vec3>, D::Error>
where
    D: Deserializer<'de>,
{
    use crate::serde_utils::Vec3Def;
    let defs: Vec<Vec3Def> = Vec::deserialize(d)?;
    Ok(defs.into_iter().map(|def| def.into()).collect())
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Clump3D {
    #[serde(serialize_with = "serialize_vec3", deserialize_with = "deserialize_vec3")]
    pub position: Vec3,
    #[serde(serialize_with = "serialize_vec3", deserialize_with = "deserialize_vec3")]
    pub velocity: Vec3,
    #[serde(serialize_with = "serialize_quat", deserialize_with = "deserialize_quat")]
    pub rotation: Quat,
    #[serde(serialize_with = "serialize_vec3", deserialize_with = "deserialize_vec3")]
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct SphereContactKey {
    pub a: usize,
    pub b: usize,
    pub ia: usize,
    pub ib: usize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlaneContactKey {
    pub clump: usize,
    pub particle: usize,
    pub axis: u8,
    pub side: i8,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdfContactKey {
    pub clump: usize,
    pub particle: usize,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterSimulation3D {
    pub templates: Vec<ClumpTemplate3D>,
    pub clumps: Vec<Clump3D>,
    #[serde(serialize_with = "serialize_vec3", deserialize_with = "deserialize_vec3")]
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
    #[serde(serialize_with = "serialize_vec3", deserialize_with = "deserialize_vec3")]
    pub bounds_min: Vec3,
    #[serde(serialize_with = "serialize_vec3", deserialize_with = "deserialize_vec3")]
    pub bounds_max: Vec3,
    #[serde(
        serialize_with = "serialize_sphere_contacts",
        deserialize_with = "deserialize_sphere_contacts"
    )]
    pub sphere_contacts: HashMap<SphereContactKey, Vec3>,
    #[serde(
        serialize_with = "serialize_plane_contacts",
        deserialize_with = "deserialize_plane_contacts"
    )]
    pub plane_contacts: HashMap<PlaneContactKey, Vec3>,
    #[serde(
        serialize_with = "serialize_sdf_contacts",
        deserialize_with = "deserialize_sdf_contacts"
    )]
    pub sdf_contacts: HashMap<SdfContactKey, Vec3>,
}

// Helpers for serializing HashMaps with glam types in values
fn serialize_sphere_contacts<S>(m: &HashMap<SphereContactKey, Vec3>, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use crate::serde_utils::Vec3Def;
    let map: HashMap<SphereContactKey, Vec3Def> = m
        .iter()
        .map(|(&k, &v)| (k, Vec3Def::from(v)))
        .collect();
    map.serialize(s)
}

fn deserialize_sphere_contacts<'de, D>(d: D) -> Result<HashMap<SphereContactKey, Vec3>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use crate::serde_utils::Vec3Def;
    let map: HashMap<SphereContactKey, Vec3Def> = HashMap::deserialize(d)?;
    Ok(map
        .into_iter()
        .map(|(k, v)| (k, Vec3::from(v)))
        .collect())
}

fn serialize_plane_contacts<S>(m: &HashMap<PlaneContactKey, Vec3>, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use crate::serde_utils::Vec3Def;
    let map: HashMap<PlaneContactKey, Vec3Def> = m
        .iter()
        .map(|(&k, &v)| (k, Vec3Def::from(v)))
        .collect();
    map.serialize(s)
}

fn deserialize_plane_contacts<'de, D>(d: D) -> Result<HashMap<PlaneContactKey, Vec3>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use crate::serde_utils::Vec3Def;
    let map: HashMap<PlaneContactKey, Vec3Def> = HashMap::deserialize(d)?;
    Ok(map
        .into_iter()
        .map(|(k, v)| (k, Vec3::from(v)))
        .collect())
}

fn serialize_sdf_contacts<S>(m: &HashMap<SdfContactKey, Vec3>, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use crate::serde_utils::Vec3Def;
    let map: HashMap<SdfContactKey, Vec3Def> = m
        .iter()
        .map(|(&k, &v)| (k, Vec3Def::from(v)))
        .collect();
    map.serialize(s)
}

fn deserialize_sdf_contacts<'de, D>(d: D) -> Result<HashMap<SdfContactKey, Vec3>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use crate::serde_utils::Vec3Def;
    let map: HashMap<SdfContactKey, Vec3Def> = HashMap::deserialize(d)?;
    Ok(map
        .into_iter()
        .map(|(k, v)| (k, Vec3::from(v)))
        .collect())
}

impl ClusterSimulation3D {
    pub fn new(bounds_min: Vec3, bounds_max: Vec3) -> Self {
        Self {
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

                self.resolve_bounds();
                self.resolve_clump_contacts();
            }
        }
    }

    /// Step with SDF collision - particles collide with solid geometry defined by SDF
    pub fn step_with_sdf(&mut self, dt: f32, sdf_params: &SdfParams) {
        // Use substeps for stability. With stiffness k=6000 and mass m=0.025kg,
        // critical timestep is ~0.004s. Using 4 substeps gives sub_dt ~0.004s.
        let substeps = 8;
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

                self.resolve_bounds();
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
        if self.clumps.is_empty() {
            return;
        }

        // Select friction coefficients based on wetness
        let friction = if wet {
            self.wet_friction
        } else {
            self.floor_friction
        };
        let rolling_friction = if wet {
            self.wet_rolling_friction
        } else {
            self.rolling_friction
        };

        let mut new_sdf_contacts: HashMap<SdfContactKey, Vec3> = HashMap::new();

        // SDF collision detection and response
        for (idx, clump) in self.clumps.iter_mut().enumerate() {
            let template = &self.templates[clump.template_idx];

            for (p_idx, offset) in template.local_offsets.iter().enumerate() {
                let r = clump.rotation * *offset;
                let pos = clump.position + r;
                let vel = clump.velocity + clump.angular_velocity.cross(r);
                let radius = template.particle_radius;

                // Sample SDF at particle position
                let (sdf_value, sdf_normal) = sample_sdf_with_gradient(
                    sdf_params.sdf,
                    pos,
                    sdf_params.grid_offset,
                    sdf_params.grid_width,
                    sdf_params.grid_height,
                    sdf_params.grid_depth,
                    sdf_params.cell_size,
                );

                // Check for penetration: SDF < radius means sphere penetrates solid
                let penetration = radius - sdf_value;
                if penetration > 0.0 && sdf_normal.length_squared() > 1e-6 {
                    let normal = sdf_normal.normalize();

                    // Push particle out of solid (position correction)
                    clump.position += normal * penetration * 1.01; // 1% extra to avoid re-penetration

                    // Velocity correction - remove velocity into solid + apply friction
                    // IMPORTANT: Apply velocity DELTA, not replacement, to preserve FLIP drag/settling
                    let v_n = vel.dot(normal);
                    if v_n < 0.0 {
                        // Moving into solid - bounce with restitution
                        let v_normal = normal * v_n;
                        let v_tangent = vel - v_normal;

                        // Compute collision response as delta from current velocity:
                        // - Normal: reverse and apply restitution
                        //   new_normal = -v_normal * restitution
                        //   delta_normal = new_normal - v_normal = -v_normal * (1 + restitution)
                        let delta_normal = -v_normal * (1.0 + self.restitution);

                        // - Tangent: apply friction damping
                        //   new_tangent = v_tangent * friction_damp
                        //   delta_tangent = new_tangent - v_tangent = v_tangent * (friction_damp - 1)
                        let friction_damp = 1.0 - friction * dt * 10.0;
                        let delta_tangent = v_tangent * (friction_damp.max(0.0) - 1.0);

                        // Apply velocity delta (preserves FLIP drag/settling contributions)
                        clump.velocity += delta_normal + delta_tangent;

                        // Track contact for history-based friction
                        let key = SdfContactKey {
                            clump: idx,
                            particle: p_idx,
                        };
                        let prev = self.sdf_contacts.get(&key).copied().unwrap_or(Vec3::ZERO);
                        let delta_t = prev + v_tangent * dt;
                        new_sdf_contacts.insert(key, delta_t);

                        // Apply rolling friction to angular velocity
                        if rolling_friction > 0.0 && clump.angular_velocity.length_squared() > 1e-8
                        {
                            let roll_damp = 1.0 - rolling_friction * 2.0;
                            clump.angular_velocity *= roll_damp.max(0.0);
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

        // Inter-clump collision (simple position-based resolution)
        self.resolve_dem_penetrations(2);

        self.sdf_contacts = new_sdf_contacts;
    }

    fn step_dem_internal(&mut self, dt: f32, sdf_params: Option<&SdfParams>) {
        if self.clumps.is_empty() {
            return;
        }

        let mut forces = vec![Vec3::ZERO; self.clumps.len()];
        let mut torques = vec![Vec3::ZERO; self.clumps.len()];

        for (idx, clump) in self.clumps.iter().enumerate() {
            let template = &self.templates[clump.template_idx];
            forces[idx] += self.gravity * template.mass;
        }

        let mut new_sphere_contacts: HashMap<SphereContactKey, Vec3> = HashMap::new();
        let mut new_plane_contacts: HashMap<PlaneContactKey, Vec3> = HashMap::new();
        let mut new_sdf_contacts: HashMap<SdfContactKey, Vec3> = HashMap::new();

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
                            let neighbor_key = (cx.saturating_add(dx), cy.saturating_add(dy), cz.saturating_add(dz));
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
                                    let clump_a = &self.clumps[i];
                                    let clump_b = &self.clumps[j];
                                    let template_a = &self.templates[clump_a.template_idx];
                                    let template_b = &self.templates[clump_b.template_idx];

                                    let delta = clump_b.position - clump_a.position;
                                    let max_dist =
                                        template_a.bounding_radius + template_b.bounding_radius;
                                    if delta.length_squared() > max_dist * max_dist {
                                        continue;
                                    }

                                    let contact_dist =
                                        template_a.particle_radius + template_b.particle_radius;
                                    let contact_dist_sq = contact_dist * contact_dist;
                                    let m_eff = (template_a.particle_mass
                                        * template_b.particle_mass)
                                        / (template_a.particle_mass + template_b.particle_mass);

                                    let c_n =
                                        dem_damping(self.restitution, self.normal_stiffness, m_eff);
                                    let c_t = dem_damping(
                                        self.restitution,
                                        self.tangential_stiffness,
                                        m_eff,
                                    );

                                    for (ia, offset_a) in
                                        template_a.local_offsets.iter().enumerate()
                                    {
                                        let ra = clump_a.rotation * *offset_a;
                                        let pa = clump_a.position + ra;
                                        let va =
                                            clump_a.velocity + clump_a.angular_velocity.cross(ra);

                                        for (ib, offset_b) in
                                            template_b.local_offsets.iter().enumerate()
                                        {
                                            let rb = clump_b.rotation * *offset_b;
                                            let pb = clump_b.position + rb;
                                            let vb = clump_b.velocity
                                                + clump_b.angular_velocity.cross(rb);

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
                                                    let center =
                                                        clump_b.position - clump_a.position;
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
                                            let mut fn_mag =
                                                self.normal_stiffness * penetration - c_n * v_n;
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

                                            let mut ft =
                                                -self.tangential_stiffness * delta_t - c_t * vt;
                                            let max_ft = self.friction * fn_mag;
                                            if ft.length_squared() > max_ft * max_ft {
                                                if ft.length_squared() > 1.0e-10 {
                                                    ft = ft.normalize() * max_ft;
                                                } else {
                                                    ft = Vec3::ZERO;
                                                }
                                                if self.tangential_stiffness > 0.0 {
                                                    delta_t = -(ft + c_t * vt)
                                                        / self.tangential_stiffness;
                                                }
                                            }
                                            new_sphere_contacts.insert(key, delta_t);

                                            let total = normal * fn_mag + ft;
                                            forces[i] -= total;
                                            forces[j] += total;
                                            torques[i] += ra.cross(-total);
                                            torques[j] += rb.cross(total);

                                            if self.rolling_friction > 0.0 {
                                                if clump_a.angular_velocity.length_squared()
                                                    > 1.0e-8
                                                {
                                                    let roll =
                                                        -clump_a.angular_velocity.normalize()
                                                            * (self.rolling_friction
                                                                * fn_mag
                                                                * template_a.particle_radius);
                                                    torques[i] += roll;
                                                }
                                                if clump_b.angular_velocity.length_squared()
                                                    > 1.0e-8
                                                {
                                                    let roll =
                                                        -clump_b.angular_velocity.normalize()
                                                            * (self.rolling_friction
                                                                * fn_mag
                                                                * template_b.particle_radius);
                                                    torques[j] += roll;
                                                }
                                            } // end rolling_friction if
                                        } // end ib loop
                                    } // end ia loop
                                } // end for &j
                            } // end if let Some
                        } // end for dz
                    } // end for dy
                } // end for dx
            } // end for &i
        } // end for spatial_hash

        for (idx, clump) in self.clumps.iter().enumerate() {
            let template = &self.templates[clump.template_idx];
            for (p_idx, offset) in template.local_offsets.iter().enumerate() {
                let r = clump.rotation * *offset;
                let pos = clump.position + r;
                let vel = clump.velocity + clump.angular_velocity.cross(r);
                let radius = template.particle_radius;

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
                        &mut forces,
                        &mut torques,
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
                        &mut forces,
                        &mut torques,
                        &mut new_plane_contacts,
                    );
                }

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
                        &mut forces,
                        &mut torques,
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
                        &mut forces,
                        &mut torques,
                        &mut new_plane_contacts,
                    );
                }

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
                        &mut forces,
                        &mut torques,
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
                        &mut forces,
                        &mut torques,
                        &mut new_plane_contacts,
                    );
                }
            }
        }

        // SDF collision - particles collide with solid geometry
        if let Some(sdf) = sdf_params {
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
                    let penetration = radius - sdf_value;
                    if penetration > 0.0 && sdf_normal.length_squared() > 1e-6 {
                        let normal = sdf_normal.normalize();

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
        }

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

        // Safety: clamp deep SDF penetration to avoid runaway forces.
        if let Some(sdf) = sdf_params {
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
                    }
                }
                let speed_sq = clump.velocity.length_squared();
                if speed_sq > max_speed * max_speed {
                    clump.velocity = clump.velocity.normalize() * max_speed;
                }
            }
        }

        // Disabled position corrections - spring-damper forces handle collision response.
        // Having both causes jitter from the two systems fighting each other.
        // self.resolve_dem_penetrations(6);
        // self.resolve_bounds_positions();

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
        let slop = 0.001;
        let percent = 0.75;
        let count = self.clumps.len();

        for _ in 0..iterations {
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

                    let contact_dist = template_a.particle_radius + template_b.particle_radius;
                    let contact_dist_sq = contact_dist * contact_dist;
                    let inv_mass_a = template_a.inv_mass();
                    let inv_mass_b = template_b.inv_mass();
                    let inv_mass_sum = inv_mass_a + inv_mass_b;
                    if inv_mass_sum <= 0.0 {
                        continue;
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
                            let correction_mag =
                                (penetration - slop).max(0.0) * percent / inv_mass_sum;
                            if correction_mag > 0.0 {
                                let correction = normal * correction_mag;
                                clump_a.position -= correction * inv_mass_a;
                                clump_b.position += correction * inv_mass_b;
                            }
                        }
                    }
                }
            }
        }
    }

    fn resolve_bounds_positions(&mut self) {
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
            } else if max.x > self.bounds_max.x {
                clump.position.x += self.bounds_max.x - max.x;
            }

            if min.y < self.bounds_min.y {
                clump.position.y += self.bounds_min.y - min.y;
            } else if max.y > self.bounds_max.y {
                clump.position.y += self.bounds_max.y - max.y;
            }

            if min.z < self.bounds_min.z {
                clump.position.z += self.bounds_min.z - min.z;
            } else if max.z > self.bounds_max.z {
                clump.position.z += self.bounds_max.z - max.z;
            }
        }
    }

    fn resolve_bounds(&mut self) {
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
                    clump.velocity.x *= 1.0 - self.friction;
                    clump.velocity.z *= 1.0 - self.friction;
                    clump.angular_velocity *= 0.7;
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
    // Convert world position to grid-local position
    let local_pos = world_pos - grid_offset;

    // Convert grid-local position to grid coordinates (cell indices)
    let fx = local_pos.x / cell_size;
    let fy = local_pos.y / cell_size;
    let fz = local_pos.z / cell_size;

    // Clamp to valid range
    let fx = fx.clamp(0.5, width as f32 - 1.5);
    let fy = fy.clamp(0.5, height as f32 - 1.5);
    let fz = fz.clamp(0.5, depth as f32 - 1.5);

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

    // Compute gradient via finite differences
    // Gradient points away from solid (towards increasing SDF)
    let h = 0.5; // Half cell for central differences

    let sample_at = |dx: f32, dy: f32, dz: f32| -> f32 {
        let px = (fx + dx).clamp(0.0, width as f32 - 1.0);
        let py = (fy + dy).clamp(0.0, height as f32 - 1.0);
        let pz = (fz + dz).clamp(0.0, depth as f32 - 1.0);
        let ii = px as usize;
        let jj = py as usize;
        let kk = pz as usize;
        idx(ii, jj, kk)
    };

    let grad_x = (sample_at(h, 0.0, 0.0) - sample_at(-h, 0.0, 0.0)) / (2.0 * h * cell_size);
    let grad_y = (sample_at(0.0, h, 0.0) - sample_at(0.0, -h, 0.0)) / (2.0 * h * cell_size);
    let grad_z = (sample_at(0.0, 0.0, h) - sample_at(0.0, 0.0, -h)) / (2.0 * h * cell_size);

    (sdf_value, Vec3::new(grad_x, grad_y, grad_z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_bounds() {
        let template = ClumpTemplate3D::generate(ClumpShape3D::Cube2, 0.05, 1.0);
        assert!(template.bounding_radius > 0.0);
        assert!(template.mass > 0.0);
    }

    #[test]
    fn test_clump_world_positions() {
        let template = ClumpTemplate3D::generate(ClumpShape3D::Tetra, 0.1, 1.0);
        let clump = Clump3D::new(Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO, 0);
        let pos = clump.particle_world_position(template.local_offsets[0]);
        assert!((pos - clump.position).length() > 0.0);
    }
}
