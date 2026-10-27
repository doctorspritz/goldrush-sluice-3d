//! Persistent, fixed-step mining domain. Rendering and camera state live outside this crate.
pub mod articulation;
pub mod bucket;
#[cfg(feature = "contact-history")]
pub mod contact_history;
pub mod earthworks;
pub mod intake;
pub mod loose_gold;
pub mod machine;
pub mod material;
pub mod outflow;
pub mod payload;
pub mod plant;
pub mod replay;
pub mod screen;
pub mod session;
pub mod site;
pub mod sluice;
pub mod slurry;
pub mod soil;
mod soil_contacts;
pub mod soil_response;
pub mod soil_water;
pub mod storage;
pub mod surface_bed;
pub mod tailings;
pub mod tracks;
pub mod water_balance;

pub use glam::Vec3;
pub use machine::{Excavator, MachineInput};
pub use material::Material;
pub use session::{Action, ControlMode, Session, FIXED_DT};
pub use site::Site;
pub use sluice::{GrainKind, Placement, Sluice};
