//! Explicit retained-world mechanical experiment. This is not a Session model.
//! The owner keeps machine/controller, soil custody and engine state together.
//! Environment changes are refused; no history is silently rebuilt or dropped.
use super::*;
use crate::contact_history::{clone_world, ContactHistory, Dispatcher, MAX_WORLD_BYTES};
use bincode::Options;

const MAGIC: &[u8; 8] = b"SLUICERS";
const VERSION: u16 = 1;
const HEADER: usize = 10;

#[derive(Clone, Serialize, Deserialize, PartialEq)]
struct Geometry {
    gravity: Vec3,
    heights: Option<Vec<f32>>,
    obstacles: Vec<[f32; 10]>,
}
impl Geometry {
    fn capture(env: &Environment<'_>) -> Result<Self, String> {
        if !env.gravity.is_finite() || !env.loads.is_empty() {
            return Err("Retained soil requires finite gravity and no external-load edits".into());
        }
        let obstacles: Vec<_> = env
            .obstacles
            .iter()
            .map(|o| {
                let [x, y, z] = o.center.to_array();
                let [a, b, c, d] = o.rotation.to_array();
                let [hx, hy, hz] = o.half_extents.to_array();
                [x, y, z, a, b, c, d, hx, hy, hz]
            })
            .collect();
        if env.obstacles.iter().any(|o| {
            !o.center.is_finite()
                || !o.half_extents.is_finite()
                || o.half_extents.min_element() <= 0.
                || !o.rotation.is_finite()
                || (o.rotation.length_squared() - 1.).abs() > 0.0001
        }) {
            return Err("Invalid retained obstacle geometry".into());
        }
        let heights = env.site.map(|site| {
            (0..crate::site::WIDTH.pow(2))
                .map(|i| {
                    site.world
                        .ground_height(i % crate::site::WIDTH, i / crate::site::WIDTH)
                })
                .collect::<Vec<_>>()
        });
        if heights
            .as_ref()
            .is_some_and(|h| h.iter().any(|v| !v.is_finite()))
        {
            return Err("Invalid retained terrain geometry".into());
        }
        Ok(Self {
            gravity: env.gravity,
            heights,
            obstacles,
        })
    }
}
#[derive(Serialize, Deserialize)]
struct Saved {
    // Canonical models have conditional JSON fields. Embedding their exact
    // JSON avoids bincode's incompatibility with skip_serializing_if layouts.
    machine_json: String,
    soil_json: String,
    geometry: Geometry,
    bodies: [r::RigidBodyHandle; 5],
    joints: [r::MultibodyJointHandle; 4],
    soil_handles: Vec<r::RigidBodyHandle>,
    numerical_world: Vec<u8>,
}
fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .with_limit(MAX_WORLD_BYTES as u64)
        .reject_trailing_bytes()
}
/// A complete mechanical owner, using the actual controller/contact equations.
/// All mutable state stays private; snapshots are engine-produced boundaries.
/// A numerical failure poisons this world and refuses further steps or saves.
/// The last valid canonical machine/soil remain available for diagnostics.
pub struct RetainedSoil {
    model: Option<Model>,
    machine: Excavator,
    soil: crate::soil::ActiveSoil,
    geometry: Geometry,
    dispatcher: Dispatcher,
}
impl Clone for RetainedSoil {
    fn clone(&self) -> Self {
        Self {
            model: self.model.as_ref().map(|m| Model {
                world: clone_world(&m.world),
                bodies: m.bodies,
                joints: m.joints,
                soil_handles: m.soil_handles.clone(),
                gravity: m.gravity,
            }),
            machine: self.machine.clone(),
            soil: self.soil.clone(),
            geometry: self.geometry.clone(),
            dispatcher: self.dispatcher,
        }
    }
}
impl RetainedSoil {
    pub fn new(
        machine: Excavator,
        soil: crate::soil::ActiveSoil,
        env: &Environment<'_>,
    ) -> Result<Self, String> {
        Self::with_contact_dispatcher(machine, soil, env, Dispatcher::RigidSoilVerified)
    }
    /// Explicit numerical policy for same-model contact comparisons. Both
    /// supported rigid-soil policies are stored and restored without upgrade.
    pub fn with_contact_dispatcher(
        mut machine: Excavator,
        mut soil: crate::soil::ActiveSoil,
        env: &Environment<'_>,
        dispatcher: Dispatcher,
    ) -> Result<Self, String> {
        if !matches!(
            dispatcher,
            Dispatcher::RigidSoil | Dispatcher::RigidSoilVerified
        ) {
            return Err("Retained soil requires a rigid-soil contact dispatcher".into());
        }
        let geometry = Geometry::capture(env)?;
        if machine.bucket_model() != crate::bucket::Model::MountedShellV2
            || !machine.valid()
            || !soil.valid()
            || env.site.is_some_and(|site| !soil.dry_contact(site))
        {
            return Err("Retained soil requires a valid drained mounted-shell mechanism".into());
        }
        let state = checked_state(
            &machine,
            MachineInput::default(),
            1. / machine.tracks.mechanical_hz as f32,
            env,
        )?;
        let mut model = build_model(&mut machine, env, &state, Some(&mut soil))?;
        // The retained experiment solves friction with normal contact throughout
        // the step. Cached guesses receive a local unbiased projection first.
        // Existing saved worlds keep their explicitly stored numerical policy.
        model.world.integration_parameters.friction_in_bias_pass = true;
        match dispatcher {
            Dispatcher::RigidSoil => crate::soil_contacts::restore_dispatcher(&mut model.world),
            Dispatcher::RigidSoilVerified => {
                crate::soil_contacts::restore_verified_dispatcher(&mut model.world)
            }
            Dispatcher::Default => unreachable!("policy checked above"),
        }
        Ok(Self {
            model: Some(model),
            machine,
            soil,
            geometry,
            dispatcher,
        })
    }
    pub fn machine(&self) -> &Excavator {
        &self.machine
    }
    pub fn soil(&self) -> &crate::soil::ActiveSoil {
        &self.soil
    }
    pub fn failed(&self) -> bool {
        self.model.is_none()
    }
    /// One mechanical step, not a Session tick or prescribed tool motion.
    /// Fuel, terrain changes, water and component transfers remain owner work.
    pub fn step(
        &mut self,
        input: MachineInput,
        dt: f32,
        env: &Environment<'_>,
    ) -> Result<(), String> {
        if self.model.is_none() {
            return Err("Retained numerical world is poisoned; restore a valid boundary".into());
        }
        let mut state = checked_state(&self.machine, input, dt, env)?;
        if self.geometry != Geometry::capture(env)? {
            return Err(
                "Retained environment changed; explicit geometry replacement is required".into(),
            );
        }
        if env.site.is_some_and(|site| !self.soil.dry_contact(site)) {
            return Err("Wet moving-soil contact is not coupled yet".into());
        }
        let mut machine = self.machine.clone();
        let mut soil = self.soil.clone();
        let mut model = self.model.take().unwrap();
        // Keep the live engine chart intact. Lift the canonical controller
        // reference onto that same chart before applying torque; the common
        // step wraps its public coordinates/reference together afterwards.
        // Rewriting a live joint to wrap it needlessly perturbs body poses.
        let (mb, _) = model.world.multibody_joints.get(model.joints[0]).unwrap();
        let live = mb.link(1).unwrap().joint.coords()[AXIS];
        state.reference[0] += live - state.coordinates[0];
        state.coordinates[0] = live;
        advance_model(
            &mut machine,
            input,
            dt,
            env,
            Some(&mut soil),
            state,
            &mut model,
        )?;
        if !machine.valid() || !soil.valid() {
            return Err(format!("Retained contact integration became invalid; world poisoned: machine={} articulation={} soil={} angles={:?}",
                machine.valid(), machine.articulation.as_ref().is_some_and(|a| a.valid_for(&machine)), soil.valid(), [machine.boom, machine.stick, machine.curl]));
        }
        self.machine = machine;
        self.soil = soil;
        self.model = Some(model);
        Ok(())
    }
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        let m = self
            .model
            .as_ref()
            .ok_or("Cannot save a poisoned retained world")?;
        #[cfg(feature = "contact-work-audit")]
        if m.world
            .integration_parameters
            .diagnostic_material_tangent_anchors
        {
            return Err("Unqualified diagnostic material tangent anchors cannot be saved".into());
        }
        let boundary = ContactHistory::from_world(clone_world(&m.world), self.dispatcher);
        let saved = Saved {
            machine_json: serde_json::to_string(&self.machine).map_err(|e| e.to_string())?,
            soil_json: serde_json::to_string(&self.soil).map_err(|e| e.to_string())?,
            geometry: self.geometry.clone(),
            bodies: m.bodies,
            joints: m.joints,
            soil_handles: m.soil_handles.clone(),
            numerical_world: boundary.encode()?,
        };
        let payload = codec().serialize(&saved).map_err(|e| e.to_string())?;
        let mut bytes = Vec::with_capacity(HEADER + payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }
    /// Restore engine-produced numerical state and its complete mechanical owner.
    /// Does not admit this experimental model into Session or arbitrary saves.
    pub fn decode(bytes: &[u8], env: &Environment<'_>) -> Result<Self, String> {
        if bytes.len() < HEADER
            || bytes.len() > HEADER + MAX_WORLD_BYTES
            || &bytes[..8] != MAGIC
            || u16::from_le_bytes([bytes[8], bytes[9]]) != VERSION
        {
            return Err("Unsupported retained-soil boundary".into());
        }
        let saved: Saved = codec()
            .deserialize(&bytes[HEADER..])
            .map_err(|e| e.to_string())?;
        if saved.geometry != Geometry::capture(env)? {
            return Err("Saved retained geometry does not match the environment".into());
        }
        let machine = serde_json::from_str(&saved.machine_json).map_err(|e| e.to_string())?;
        let soil = serde_json::from_str(&saved.soil_json).map_err(|e| e.to_string())?;
        let mut result = Self::new(machine, soil, env)?;
        let boundary = ContactHistory::decode(&saved.numerical_world)?;
        if !matches!(
            boundary.dispatcher(),
            Dispatcher::RigidSoil | Dispatcher::RigidSoilVerified
        ) {
            return Err("Wrong retained contact policy".into());
        }
        result.dispatcher = boundary.dispatcher();
        let world = boundary.into_world();
        let expected = result.model.as_ref().unwrap();
        if saved.bodies != expected.bodies
            || saved.joints != expected.joints
            || saved.soil_handles != expected.soil_handles
            || world.bodies.len() != expected.world.bodies.len()
            || world.colliders.len() != expected.world.colliders.len()
            || world.multibody_joints.iter().count()
                != expected.world.multibody_joints.iter().count()
            || world.impulse_joints.len() != expected.world.impulse_joints.len()
        {
            return Err("Saved retained handle maps do not match the owner".into());
        }
        let mut parameters = expected.world.integration_parameters;
        parameters.dt = world.integration_parameters.dt;
        // Both stored policies are supported: original relax-only friction,
        // and the new full-friction projected warm start. All other model
        // parameters must still match exactly; decoding never upgrades a save.
        parameters.friction_in_bias_pass = world.integration_parameters.friction_in_bias_pass;
        if !parameters.dt.is_finite()
            || parameters.dt <= 0.
            || parameters.dt > 0.05
            || world.gravity != expected.world.gravity
            || bincode::serialize(&parameters).map_err(|e| e.to_string())?
                != bincode::serialize(&world.integration_parameters).map_err(|e| e.to_string())?
        {
            return Err("Saved retained parameters do not match the model".into());
        }
        for (h, c) in expected.world.colliders.iter() {
            let Some(other) = world.colliders.get(h) else {
                return Err("Missing retained collider".into());
            };
            let local = c.position_wrt_parent();
            let other_local = other.position_wrt_parent();
            let pose_matches = match (local, other_local) {
                (None, None) => true,
                (Some(a), Some(b)) => poses_match(a, b),
                _ => false,
            };
            if c.parent() != other.parent()
                || c.user_data != other.user_data
                || !pose_matches
                || c.friction() != other.friction()
                || c.restitution() != other.restitution()
                || bincode::serialize(c.shared_shape()).map_err(|e| e.to_string())?
                    != bincode::serialize(other.shared_shape()).map_err(|e| e.to_string())?
            {
                return Err("Retained collider geometry/material does not match its owner".into());
            }
        }
        for (h, b) in expected.world.bodies.iter() {
            let Some(other) = world.bodies.get(h) else {
                return Err("Missing retained body".into());
            };
            if b.body_type() != other.body_type()
                || !poses_match(b.position(), other.position())
                || !other.linvel().is_finite()
                || !other.angvel().is_finite()
                || (b.linvel() - other.linvel()).length() > 0.01
                || (b.angvel() - other.angvel()).length() > 0.01
                || (b.mass() - other.mass()).abs() > 0.00001 * b.mass().max(1.)
            {
                return Err("Retained numerical body does not match its canonical owner".into());
            }
        }
        let model = result.model.as_mut().unwrap();
        model.world = world;
        Ok(result)
    }
}
fn poses_match(a: &r::Pose, b: &r::Pose) -> bool {
    let qa = sq(a.rotation);
    let qb = sq(b.rotation);
    (a.translation - b.translation).length() < 0.001
        && (qa - qb).length().min((qa + qb).length()) < 0.0002
}

#[cfg(test)]
mod tests;
