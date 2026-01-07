use glam::Vec2;

/// Player input state for panning controls.
#[derive(Clone, Copy, Debug, Default)]
pub struct PanInput {
    /// Pan tilt angle in radians (x: left/right, y: forward/back).
    pub tilt: Vec2,
    /// Swirl speed in RPM.
    pub swirl_rpm: f32,
    /// Add water action.
    pub add_water: bool,
    /// Shake pan action.
    pub shake: bool,
    /// Dump contents action.
    pub dump: bool,
}

impl PanInput {
    /// Maximum tilt angle in radians (~30 degrees).
    pub const MAX_TILT: f32 = 0.52;

    pub fn clamp_tilt(&mut self) {
        self.tilt.x = self.tilt.x.clamp(-Self::MAX_TILT, Self::MAX_TILT);
        self.tilt.y = self.tilt.y.clamp(-Self::MAX_TILT, Self::MAX_TILT);
    }
}
