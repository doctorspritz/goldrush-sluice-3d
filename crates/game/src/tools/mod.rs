pub mod shovel;

pub use shovel::Shovel;
use glam::Vec3;

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

#[derive(Clone, Copy, Debug)]
pub struct ToolSpec {
    pub kind: ToolKind,
    pub radius: f32,
    pub depth: f32,
    pub strength: f32,
    pub cooldown: f32,
}

impl ToolSpec {
    pub fn to_sim_tool(&self) -> Tool {
        Tool {
            kind: self.kind,
            radius: self.radius,
            depth: self.depth,
            strength: self.strength,
        }
    }
}
