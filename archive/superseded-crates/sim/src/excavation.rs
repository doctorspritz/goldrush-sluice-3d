use glam::Vec3;

use crate::terrain::TerrainMaterial;

#[derive(Clone, Copy, Debug)]
pub enum ToolKind {
    Shovel,
    Pickaxe,
}

/// Tool parameters used for excavation.
#[derive(Clone, Copy, Debug)]
pub struct Tool {
    pub kind: ToolKind,
    pub radius: f32,
    pub depth: f32,
    pub strength: f32,
}

impl Tool {
    pub fn shovel() -> Self {
        Self {
            kind: ToolKind::Shovel,
            radius: 0.3,
            depth: 0.1,
            strength: 3.0,
        }
    }

    pub fn pickaxe() -> Self {
        Self {
            kind: ToolKind::Pickaxe,
            radius: 0.2,
            depth: 0.15,
            strength: 10.0,
        }
    }
}

/// Excavated material chunk produced by digging.
#[derive(Clone, Copy, Debug)]
pub struct ExcavatedParticle {
    pub position: Vec3,
    pub material: TerrainMaterial,
    pub volume: f32,
    pub gold_ppm: f32,
}
