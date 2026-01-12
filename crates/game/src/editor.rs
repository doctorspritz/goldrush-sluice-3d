//! Washplant Editor - Layout and piece definitions
//!
//! Provides data structures for the washplant editor, including:
//! - GutterPiece: Angled U-channel for water collection
//! - SluicePiece: Recovery sluice with riffles
//! - EditorLayout: Container for all pieces

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Rotation around Y axis (0, 90, 180, 270 degrees)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum Rotation {
    #[default]
    R0 = 0,
    R90 = 1,
    R180 = 2,
    R270 = 3,
}

impl Rotation {
    pub fn degrees(&self) -> f32 {
        match self {
            Rotation::R0 => 0.0,
            Rotation::R90 => 90.0,
            Rotation::R180 => 180.0,
            Rotation::R270 => 270.0,
        }
    }

    pub fn radians(&self) -> f32 {
        self.degrees().to_radians()
    }

    pub fn next(&self) -> Self {
        match self {
            Rotation::R0 => Rotation::R90,
            Rotation::R90 => Rotation::R180,
            Rotation::R180 => Rotation::R270,
            Rotation::R270 => Rotation::R0,
        }
    }

    pub fn prev(&self) -> Self {
        match self {
            Rotation::R0 => Rotation::R270,
            Rotation::R90 => Rotation::R0,
            Rotation::R180 => Rotation::R90,
            Rotation::R270 => Rotation::R180,
        }
    }
}

/// An angled gutter piece - U-channel that collects water
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GutterPiece {
    /// Unique ID for selection
    pub id: u32,
    /// Position in world space (meters)
    #[serde(with = "vec3_serde")]
    pub position: Vec3,
    /// Rotation around Y axis
    pub rotation: Rotation,
    /// Tilt angle for flow direction (-45 to 45 degrees)
    pub angle_deg: f32,
    /// Total length (meters)
    pub length: f32,
    /// Channel width at inlet/start (meters)
    pub width: f32,
    /// Channel width at outlet/end (meters) - for funnel effect
    #[serde(default = "default_gutter_end_width")]
    pub end_width: f32,
    /// Side wall height (meters)
    pub wall_height: f32,
}

fn default_gutter_end_width() -> f32 {
    0.3 // Same as default width
}

impl Default for GutterPiece {
    fn default() -> Self {
        Self {
            id: 0,
            // Default at grid-aligned position (multiples of 0.1m for MOVE_STEP alignment)
            position: Vec3::new(0.0, 0.9, 0.0),
            rotation: Rotation::R0,
            angle_deg: 10.0,
            length: 1.0,
            width: 0.3,
            end_width: 0.3,
            wall_height: 0.1,
        }
    }
}

impl GutterPiece {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }

    /// Height difference from inlet (high) to outlet (low) based on angle
    pub fn height_drop(&self) -> f32 {
        self.length * self.angle_deg.to_radians().tan()
    }

    /// Get inlet position (high end)
    pub fn inlet_position(&self) -> Vec3 {
        let half_drop = self.height_drop() / 2.0;
        let offset = match self.rotation {
            Rotation::R0 => Vec3::new(-self.length / 2.0, half_drop, 0.0),
            Rotation::R90 => Vec3::new(0.0, half_drop, -self.length / 2.0),
            Rotation::R180 => Vec3::new(self.length / 2.0, half_drop, 0.0),
            Rotation::R270 => Vec3::new(0.0, half_drop, self.length / 2.0),
        };
        self.position + offset
    }

    /// Get outlet position (low end)
    pub fn outlet_position(&self) -> Vec3 {
        let half_drop = self.height_drop() / 2.0;
        let offset = match self.rotation {
            Rotation::R0 => Vec3::new(self.length / 2.0, -half_drop, 0.0),
            Rotation::R90 => Vec3::new(0.0, -half_drop, self.length / 2.0),
            Rotation::R180 => Vec3::new(-self.length / 2.0, -half_drop, 0.0),
            Rotation::R270 => Vec3::new(0.0, -half_drop, -self.length / 2.0),
        };
        self.position + offset
    }

    /// Move position by delta
    pub fn translate(&mut self, delta: Vec3) {
        self.position += delta;
    }

    /// Rotate 90 degrees clockwise
    pub fn rotate_cw(&mut self) {
        self.rotation = self.rotation.next();
    }

    /// Rotate 90 degrees counter-clockwise
    pub fn rotate_ccw(&mut self) {
        self.rotation = self.rotation.prev();
    }

    /// Adjust angle within bounds
    pub fn adjust_angle(&mut self, delta: f32) {
        self.angle_deg = (self.angle_deg + delta).clamp(-45.0, 45.0);
    }

    /// Adjust length within bounds
    pub fn adjust_length(&mut self, delta: f32) {
        self.length = (self.length + delta).clamp(0.2, 5.0);
    }

    /// Adjust start width within bounds (same range as sluice for matching)
    pub fn adjust_width(&mut self, delta: f32) {
        let raw = self.width + delta;
        // Round to nearest 0.05 to avoid float drift
        self.width = (raw * 20.0).round() / 20.0;
        self.width = self.width.clamp(0.15, 1.5);
    }

    /// Adjust end width within bounds
    pub fn adjust_end_width(&mut self, delta: f32) {
        let raw = self.end_width + delta;
        // Round to nearest 0.05 to avoid float drift
        self.end_width = (raw * 20.0).round() / 20.0;
        self.end_width = self.end_width.clamp(0.15, 1.5);
    }

    /// Get width at a position along the gutter (0.0 = inlet, 1.0 = outlet)
    pub fn width_at(&self, t: f32) -> f32 {
        self.width + (self.end_width - self.width) * t.clamp(0.0, 1.0)
    }

    /// Maximum width (for grid sizing)
    pub fn max_width(&self) -> f32 {
        self.width.max(self.end_width)
    }
}

/// A sluice piece - recovery sluice with riffles
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SluicePiece {
    /// Unique ID for selection
    pub id: u32,
    /// Position in world space (meters)
    #[serde(with = "vec3_serde")]
    pub position: Vec3,
    /// Rotation around Y axis
    pub rotation: Rotation,
    /// Total length (meters)
    pub length: f32,
    /// Channel width (meters)
    pub width: f32,
    /// Floor slope angle (degrees)
    pub slope_deg: f32,
    /// Spacing between riffles (meters)
    pub riffle_spacing: f32,
    /// Riffle height (meters)
    pub riffle_height: f32,
}

impl Default for SluicePiece {
    fn default() -> Self {
        Self {
            id: 0,
            // Default at grid-aligned position (multiples of 0.1m for MOVE_STEP alignment)
            // Positioned to potentially receive water from default gutter outlet
            position: Vec3::new(1.1, 0.5, 0.0),
            rotation: Rotation::R0,
            length: 2.0,
            width: 0.3, // Same default as gutter for easy matching
            slope_deg: 8.0,
            riffle_spacing: 0.15,
            riffle_height: 0.02,
        }
    }
}

impl SluicePiece {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }

    /// Height difference from inlet to outlet
    pub fn height_drop(&self) -> f32 {
        self.length * self.slope_deg.to_radians().tan()
    }

    /// Get inlet position (high end)
    pub fn inlet_position(&self) -> Vec3 {
        let half_drop = self.height_drop() / 2.0;
        let offset = match self.rotation {
            Rotation::R0 => Vec3::new(-self.length / 2.0, half_drop, 0.0),
            Rotation::R90 => Vec3::new(0.0, half_drop, -self.length / 2.0),
            Rotation::R180 => Vec3::new(self.length / 2.0, half_drop, 0.0),
            Rotation::R270 => Vec3::new(0.0, half_drop, self.length / 2.0),
        };
        self.position + offset
    }

    /// Get outlet position (low end)
    pub fn outlet_position(&self) -> Vec3 {
        let half_drop = self.height_drop() / 2.0;
        let offset = match self.rotation {
            Rotation::R0 => Vec3::new(self.length / 2.0, -half_drop, 0.0),
            Rotation::R90 => Vec3::new(0.0, -half_drop, self.length / 2.0),
            Rotation::R180 => Vec3::new(-self.length / 2.0, -half_drop, 0.0),
            Rotation::R270 => Vec3::new(0.0, -half_drop, -self.length / 2.0),
        };
        self.position + offset
    }

    /// Move position by delta
    pub fn translate(&mut self, delta: Vec3) {
        self.position += delta;
    }

    /// Rotate 90 degrees clockwise
    pub fn rotate_cw(&mut self) {
        self.rotation = self.rotation.next();
    }

    /// Rotate 90 degrees counter-clockwise
    pub fn rotate_ccw(&mut self) {
        self.rotation = self.rotation.prev();
    }

    /// Adjust slope within bounds
    pub fn adjust_slope(&mut self, delta: f32) {
        self.slope_deg = (self.slope_deg + delta).clamp(2.0, 20.0);
    }

    /// Adjust length within bounds
    pub fn adjust_length(&mut self, delta: f32) {
        self.length = (self.length + delta).clamp(0.5, 10.0);
    }

    /// Adjust width within bounds (same range as gutter for matching)
    pub fn adjust_width(&mut self, delta: f32) {
        let raw = self.width + delta;
        // Round to nearest 0.05 to avoid float drift
        self.width = (raw * 20.0).round() / 20.0;
        self.width = self.width.clamp(0.15, 1.5);
    }
}

/// A water emitter piece - spawns water particles
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmitterPiece {
    /// Unique ID for selection
    pub id: u32,
    /// Position in world space (meters)
    #[serde(with = "vec3_serde")]
    pub position: Vec3,
    /// Rotation around Y axis
    pub rotation: Rotation,
    /// Emission rate (particles per second)
    pub rate: f32,
    /// Emission spread angle (degrees)
    pub spread_deg: f32,
    /// Initial velocity magnitude (m/s)
    pub velocity: f32,
    /// Emitter radius (meters)
    pub radius: f32,
    /// Spray bar width (meters) - particles spread across this width
    #[serde(default = "default_emitter_width")]
    pub width: f32,
}

fn default_emitter_width() -> f32 {
    0.3
}

impl Default for EmitterPiece {
    fn default() -> Self {
        Self {
            id: 0,
            // Default at grid-aligned position (multiples of 0.1m for MOVE_STEP alignment)
            // Positioned above default gutter inlet area
            position: Vec3::new(-0.2, 1.2, 0.0),
            rotation: Rotation::R0,
            rate: 100.0,
            spread_deg: 15.0,
            velocity: 1.0,
            radius: 0.1,
            width: 0.3,
        }
    }
}

impl EmitterPiece {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }

    /// Get emission direction based on rotation
    pub fn emit_direction(&self) -> Vec3 {
        match self.rotation {
            Rotation::R0 => Vec3::new(1.0, 0.0, 0.0),
            Rotation::R90 => Vec3::new(0.0, 0.0, 1.0),
            Rotation::R180 => Vec3::new(-1.0, 0.0, 0.0),
            Rotation::R270 => Vec3::new(0.0, 0.0, -1.0),
        }
    }

    /// Move position by delta
    pub fn translate(&mut self, delta: Vec3) {
        self.position += delta;
    }

    /// Rotate 90 degrees clockwise
    pub fn rotate_cw(&mut self) {
        self.rotation = self.rotation.next();
    }

    /// Rotate 90 degrees counter-clockwise
    pub fn rotate_ccw(&mut self) {
        self.rotation = self.rotation.prev();
    }

    /// Adjust emission rate within bounds
    pub fn adjust_rate(&mut self, delta: f32) {
        self.rate = (self.rate + delta).clamp(10.0, 1000.0);
    }

    /// Adjust velocity within bounds
    pub fn adjust_velocity(&mut self, delta: f32) {
        self.velocity = (self.velocity + delta).clamp(0.1, 10.0);
    }

    /// Adjust spread angle within bounds
    pub fn adjust_spread(&mut self, delta: f32) {
        self.spread_deg = (self.spread_deg + delta).clamp(0.0, 90.0);
    }

    /// Adjust spray bar width within bounds (same range as gutter/sluice)
    pub fn adjust_width(&mut self, delta: f32) {
        let raw = self.width + delta;
        // Round to nearest 0.05 to avoid float drift
        self.width = (raw * 20.0).round() / 20.0;
        self.width = self.width.clamp(0.15, 1.5);
    }
}

/// Piece type enum for selection
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieceType {
    Gutter,
    Sluice,
    Emitter,
}

/// Selection state
#[derive(Clone, Copy, Debug, Default)]
pub enum Selection {
    #[default]
    None,
    Gutter(usize),
    Sluice(usize),
    Emitter(usize),
}

impl Selection {
    pub fn is_some(&self) -> bool {
        !matches!(self, Selection::None)
    }
}

/// Editor mode
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EditorMode {
    #[default]
    Select,
    PlaceGutter,
    PlaceSluice,
    PlaceEmitter,
}

/// Editor layout containing all pieces
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EditorLayout {
    pub gutters: Vec<GutterPiece>,
    pub sluices: Vec<SluicePiece>,
    pub emitters: Vec<EmitterPiece>,
    next_id: u32,
}

impl EditorLayout {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a pre-connected gutter + sluice layout
    /// The gutter outlet connects directly to the sluice inlet (Y and Z aligned)
    pub fn new_connected() -> Self {
        let mut layout = Self::default();

        // Create gutter at a reasonable starting position
        let gutter_id = layout.next_id();
        let mut gutter = GutterPiece::new(gutter_id);
        gutter.position = Vec3::new(0.0, 0.9, 0.0);
        gutter.rotation = Rotation::R0;

        // Create sluice - position it so inlet matches gutter outlet
        let sluice_id = layout.next_id();
        let mut sluice = SluicePiece::new(sluice_id);
        sluice.rotation = Rotation::R0;

        // Calculate connection point:
        // Gutter outlet = gutter.position + (length/2, -half_drop, 0) for R0
        // Sluice inlet = sluice.position + (-length/2, +half_drop, 0) for R0
        // So: sluice.position = gutter.outlet - (-length/2, +half_drop, 0)
        //                     = gutter.outlet + (length/2, -half_drop, 0)
        let gutter_outlet = gutter.outlet_position();
        let sluice_half_drop = sluice.height_drop() / 2.0;

        sluice.position = Vec3::new(
            gutter_outlet.x + sluice.length / 2.0, // X: outlet + half sluice length
            gutter_outlet.y - sluice_half_drop,    // Y: outlet Y - sluice half_drop
            gutter_outlet.z,                        // Z: same as gutter
        );

        // Add emitter above gutter inlet
        let emitter_id = layout.next_id();
        let mut emitter = EmitterPiece::new(emitter_id);
        let gutter_inlet = gutter.inlet_position();
        emitter.position = Vec3::new(gutter_inlet.x, gutter_inlet.y + 0.15, gutter_inlet.z);
        emitter.rotation = Rotation::R0;

        layout.gutters.push(gutter);
        layout.sluices.push(sluice);
        layout.emitters.push(emitter);

        layout
    }

    pub fn next_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Add a new gutter at position
    pub fn add_gutter(&mut self, position: Vec3) -> usize {
        let id = self.next_id();
        let mut gutter = GutterPiece::new(id);
        gutter.position = position;
        self.gutters.push(gutter);
        self.gutters.len() - 1
    }

    /// Add a new sluice at position
    pub fn add_sluice(&mut self, position: Vec3) -> usize {
        let id = self.next_id();
        let mut sluice = SluicePiece::new(id);
        sluice.position = position;
        self.sluices.push(sluice);
        self.sluices.len() - 1
    }

    /// Remove a gutter by index
    pub fn remove_gutter(&mut self, index: usize) {
        if index < self.gutters.len() {
            self.gutters.remove(index);
        }
    }

    /// Remove a sluice by index
    pub fn remove_sluice(&mut self, index: usize) {
        if index < self.sluices.len() {
            self.sluices.remove(index);
        }
    }

    /// Add a new emitter at position
    pub fn add_emitter(&mut self, position: Vec3) -> usize {
        let id = self.next_id();
        let mut emitter = EmitterPiece::new(id);
        emitter.position = position;
        self.emitters.push(emitter);
        self.emitters.len() - 1
    }

    /// Remove an emitter by index
    pub fn remove_emitter(&mut self, index: usize) {
        if index < self.emitters.len() {
            self.emitters.remove(index);
        }
    }

    /// Get total piece count
    pub fn piece_count(&self) -> usize {
        self.gutters.len() + self.sluices.len() + self.emitters.len()
    }

    /// Save layout to JSON file
    pub fn save_json(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load layout from JSON file
    pub fn load_json(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let layout = serde_json::from_str(&json)?;
        Ok(layout)
    }
}

/// Custom serde module for Vec3
mod vec3_serde {
    use glam::Vec3;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct Vec3Repr {
        x: f32,
        y: f32,
        z: f32,
    }

    pub fn serialize<S>(vec: &Vec3, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Vec3Repr {
            x: vec.x,
            y: vec.y,
            z: vec.z,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec3, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = Vec3Repr::deserialize(deserializer)?;
        Ok(Vec3::new(repr.x, repr.y, repr.z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gutter_defaults() {
        let gutter = GutterPiece::default();
        assert_eq!(gutter.angle_deg, 10.0);
        assert_eq!(gutter.length, 1.0);
        assert_eq!(gutter.width, 0.3);
    }

    #[test]
    fn test_gutter_rotation() {
        let mut gutter = GutterPiece::default();
        assert_eq!(gutter.rotation, Rotation::R0);
        gutter.rotate_cw();
        assert_eq!(gutter.rotation, Rotation::R90);
        gutter.rotate_cw();
        assert_eq!(gutter.rotation, Rotation::R180);
        gutter.rotate_ccw();
        assert_eq!(gutter.rotation, Rotation::R90);
    }

    #[test]
    fn test_gutter_angle_bounds() {
        let mut gutter = GutterPiece::default();
        gutter.adjust_angle(100.0);
        assert_eq!(gutter.angle_deg, 45.0);
        gutter.adjust_angle(-100.0);
        assert_eq!(gutter.angle_deg, -45.0);
    }

    #[test]
    fn test_sluice_defaults() {
        let sluice = SluicePiece::default();
        assert_eq!(sluice.length, 2.0);
        assert_eq!(sluice.slope_deg, 8.0);
    }

    #[test]
    fn test_layout_add_pieces() {
        let mut layout = EditorLayout::new();
        let g_idx = layout.add_gutter(Vec3::new(0.0, 1.0, 0.0));
        let s_idx = layout.add_sluice(Vec3::new(2.0, 0.0, 0.0));

        assert_eq!(layout.gutters.len(), 1);
        assert_eq!(layout.sluices.len(), 1);
        assert_eq!(g_idx, 0);
        assert_eq!(s_idx, 0);
        assert_eq!(layout.piece_count(), 2);
    }

    #[test]
    fn test_layout_serialization() {
        let mut layout = EditorLayout::new();
        layout.add_gutter(Vec3::new(1.0, 2.0, 3.0));
        layout.gutters[0].angle_deg = 15.0;
        layout.gutters[0].rotation = Rotation::R90;

        let json = serde_json::to_string(&layout).unwrap();
        let loaded: EditorLayout = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.gutters.len(), 1);
        assert_eq!(loaded.gutters[0].angle_deg, 15.0);
        assert_eq!(loaded.gutters[0].rotation, Rotation::R90);
    }
}
