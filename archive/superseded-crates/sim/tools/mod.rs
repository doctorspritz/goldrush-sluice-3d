pub mod shovel;

pub use shovel::Shovel;
pub use sim::excavation::ToolKind;

use sim::excavation::Tool;

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
