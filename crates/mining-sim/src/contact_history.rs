//! Numerical rigid-world boundaries for contact-lifecycle validation.
//!
//! Application owners must persist their body/joint handle maps alongside this
//! world. Session saves do not admit this state yet: persistent integration and
//! its physical/convergence checks remain separate from this storage boundary.
use bincode::Options;
use rapier3d::{parry::query::DefaultQueryDispatcher, prelude as r};

const MAGIC: &[u8; 8] = b"SLUICEWH";
const VERSION: u16 = 1;
const HEADER: usize = 11;
/// Resource refusal, never a reduction of admitted physical geometry.
pub const MAX_WORLD_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dispatcher {
    Default,
    RigidSoil,
    RigidSoilVerified,
}
impl Dispatcher {
    fn install(self, world: &mut r::PhysicsWorld) {
        match self {
            Self::Default => world
                .narrow_phase
                .set_query_dispatcher(DefaultQueryDispatcher),
            Self::RigidSoil => crate::soil_contacts::restore_dispatcher(world),
            Self::RigidSoilVerified => crate::soil_contacts::restore_verified_dispatcher(world),
        }
    }
    fn code(self) -> u8 {
        match self {
            Self::Default => 0,
            Self::RigidSoil => 1,
            Self::RigidSoilVerified => 2,
        }
    }
}
/// Owned numerical state. Cloning preserves solver state while rebuilding the
/// derived pipeline/CCD workspaces, just as binary restoration does.
pub struct ContactHistory {
    world: r::PhysicsWorld,
    dispatcher: Dispatcher,
}
fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .with_limit(MAX_WORLD_BYTES as u64)
        .reject_trailing_bytes()
}
impl ContactHistory {
    /// Select an explicitly supported query dispatcher. All physical integration
    /// and contact-softness parameters in the supplied world remain intact.
    pub fn from_world(mut world: r::PhysicsWorld, dispatcher: Dispatcher) -> Self {
        dispatcher.install(&mut world);
        Self { world, dispatcher }
    }
    pub fn world(&self) -> &r::PhysicsWorld {
        &self.world
    }
    pub fn world_mut(&mut self) -> &mut r::PhysicsWorld {
        &mut self.world
    }
    pub fn into_world(self) -> r::PhysicsWorld {
        self.world
    }
    pub fn dispatcher(&self) -> Dispatcher {
        self.dispatcher
    }
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        #[cfg(feature = "contact-work-audit")]
        if self
            .world
            .integration_parameters
            .diagnostic_material_tangent_anchors
        {
            return Err("Unqualified diagnostic material tangent anchors cannot be saved".into());
        }
        #[cfg(feature = "contact-work-audit")]
        if self
            .world
            .integration_parameters
            .diagnostic_preserve_normal_compliance
        {
            return Err("Unqualified diagnostic normal compliance cannot be saved".into());
        }
        let payload = codec().serialize(&self.world).map_err(|e| e.to_string())?;
        let mut bytes = Vec::with_capacity(HEADER + payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.push(self.dispatcher.code());
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }
    /// Schema/codec boundary for engine-produced state. Application-level scene,
    /// custody and handle-map validation is required before Session admission.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER || bytes.len() > MAX_WORLD_BYTES + HEADER {
            return Err("Invalid contact-history length".into());
        }
        if &bytes[..8] != MAGIC || u16::from_le_bytes([bytes[8], bytes[9]]) != VERSION {
            return Err("Unsupported contact-history schema".into());
        }
        let dispatcher = match bytes[10] {
            0 => Dispatcher::Default,
            1 => Dispatcher::RigidSoil,
            2 => Dispatcher::RigidSoilVerified,
            _ => return Err("Unsupported contact dispatcher".into()),
        };
        let world = codec()
            .deserialize(&bytes[HEADER..])
            .map_err(|e| e.to_string())?;
        Ok(Self::from_world(world, dispatcher))
    }
}
impl Clone for ContactHistory {
    fn clone(&self) -> Self {
        let world = clone_world(&self.world);
        Self {
            world,
            dispatcher: self.dispatcher,
        }
    }
}

/// Copy numerical sets; derived pipeline/CCD workspaces start empty.
pub(crate) fn clone_world(w: &r::PhysicsWorld) -> r::PhysicsWorld {
    r::PhysicsWorld {
        gravity: w.gravity,
        integration_parameters: w.integration_parameters,
        islands: w.islands.clone(),
        broad_phase: w.broad_phase.clone(),
        narrow_phase: w.narrow_phase.clone(),
        bodies: w.bodies.clone(),
        colliders: w.colliders.clone(),
        impulse_joints: w.impulse_joints.clone(),
        multibody_joints: w.multibody_joints.clone(),
        soft_bodies: w.soft_bodies.clone(),
        ..Default::default()
    }
}
