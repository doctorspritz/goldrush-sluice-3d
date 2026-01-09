use super::{ToolKind, ToolSpec};

#[derive(Clone, Copy, Debug)]
pub struct Shovel {
    spec: ToolSpec,
}

impl Shovel {
    pub fn new() -> Self {
        Self {
            spec: ToolSpec {
                kind: ToolKind::Shovel,
                radius: 0.3,
                depth: 0.1,
                strength: 3.0,
                cooldown: 0.3,
            },
        }
    }

    pub fn spec(&self) -> ToolSpec {
        self.spec
    }
}
