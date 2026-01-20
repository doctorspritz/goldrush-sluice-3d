//! Scenario system for saving and restoring simulation states.
//!
//! A Scenario combines the static geometry (EditorLayout) with the dynamic
//! state of particles and their velocities (SimulationState).

use serde::{Deserialize, Serialize};
use sim3d::serde_utils::{deserialize_vec3, serialize_vec3};
use sim3d::{ClusterSimulation3D, FlipSimulation3D, Vec3};
use crate::editor::EditorLayout;

/// A snapshot of the dynamic simulation state.
/// Includes one or more FLIP fluid particles grids and DEM clumps.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationState {
    /// Fluid simulation states (particles and grid)
    #[serde(default)]
    pub flips: Vec<FlipSimulation3D>,
    /// DEM solid simulation state (clumps and contacts)
    pub dem: Option<ClusterSimulation3D>,
}

/// A complete simulation scenario including geometry and dynamic state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scenario {
    /// Static geometry and emitters
    pub layout: EditorLayout,
    /// Optional dynamic state to restore
    pub state: Option<SimulationState>,
    /// Metadata about the scenario
    pub name: String,
    pub description: String,
    /// Optional camera override
    #[serde(
        default,
        serialize_with = "serialize_opt_vec3",
        deserialize_with = "deserialize_opt_vec3"
    )]
    pub camera_target: Option<Vec3>,
    pub camera_distance: Option<f32>,
}

fn serialize_opt_vec3<S>(v: &Option<Vec3>, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match v {
        Some(vec) => serialize_vec3(vec, s),
        None => s.serialize_none(),
    }
}

fn deserialize_opt_vec3<'de, D>(d: D) -> Result<Option<Vec3>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<sim3d::serde_utils::Vec3Def>::deserialize(d).map(|opt| opt.map(Vec3::from))
}

impl Scenario {
    pub fn new(name: &str, layout: EditorLayout) -> Self {
        Self {
            layout,
            state: None,
            name: name.to_string(),
            description: String::new(),
            camera_target: None,
            camera_distance: None,
        }
    }

    /// Save scenario to a JSON file.
    pub fn save_json(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load scenario from a JSON file.
    pub fn load_json(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let scenario = serde_json::from_str(&json)?;
        Ok(scenario)
    }
}
