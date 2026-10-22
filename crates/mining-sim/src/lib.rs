//! Persistent, fixed-step mining domain. Rendering and camera state live outside this crate.
pub mod earthworks;
pub mod machine;
pub mod material;
pub mod outflow;
pub mod replay;
pub mod session;
pub mod site;
pub mod sluice;
pub mod slurry;
pub mod storage;
pub mod water_balance;

pub use glam::Vec3;
pub use machine::{Excavator, MachineInput};
pub use material::Material;
pub use session::{Action, ControlMode, Session, FIXED_DT};
pub use site::Site;
pub use sluice::{GrainKind, Placement, Sluice};
