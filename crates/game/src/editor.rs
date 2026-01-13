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

    /// Adjust wall height within bounds
    pub fn adjust_wall_height(&mut self, delta: f32) {
        let raw = self.wall_height + delta;
        self.wall_height = (raw * 100.0).round() / 100.0;
        self.wall_height = self.wall_height.clamp(0.02, 0.5);
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
    #[serde(default = "default_spawn_sediment")]
    pub spawn_sediment: bool,
}

fn default_emitter_width() -> f32 {
    0.3
}

fn default_spawn_sediment() -> bool {
    true
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
            spawn_sediment: true,
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

/// A shaker deck - horizontal screen/grid for particle size separation
/// Small particles (≤ hole_size) pass through, larger ones slide off
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShakerDeckPiece {
    /// Unique ID for selection
    pub id: u32,
    /// Position in world space (meters) - center of deck
    #[serde(with = "vec3_serde")]
    pub position: Vec3,
    /// Rotation around Y axis
    pub rotation: Rotation,
    /// Deck length along flow direction (meters)
    pub length: f32,
    /// Deck width at inlet (meters)
    pub width: f32,
    /// Deck width at outlet (meters) - for funnel effect
    #[serde(default = "default_shaker_end_width")]
    pub end_width: f32,
    /// Slight tilt angle for material to slide off (degrees, typically 5-15)
    pub tilt_deg: f32,
    /// Grid hole size (meters) - particles smaller than this pass through
    pub hole_size: f32,
    /// Side wall height (meters)
    pub wall_height: f32,
    /// Bar/wire thickness (meters) - the solid parts of the grid
    pub bar_thickness: f32,
}

fn default_shaker_end_width() -> f32 {
    0.5 // Same as default width
}

impl Default for ShakerDeckPiece {
    fn default() -> Self {
        Self {
            id: 0,
            position: Vec3::new(0.0, 1.2, 0.0),
            rotation: Rotation::R0,
            length: 0.8,
            width: 0.5,
            end_width: 0.5,
            tilt_deg: 8.0,    // Slight tilt for material flow
            hole_size: 0.005, // 5mm holes
            wall_height: 0.08,
            bar_thickness: 0.003, // 3mm bars
        }
    }
}

impl ShakerDeckPiece {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }

    /// Height drop from inlet to outlet based on tilt
    pub fn height_drop(&self) -> f32 {
        self.length * self.tilt_deg.to_radians().tan()
    }

    /// Get inlet position (high end where material enters)
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

    /// Get outlet position (low end where oversize exits)
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

    /// Get underside position (where fines fall to) - below deck center
    pub fn underside_position(&self) -> Vec3 {
        Vec3::new(self.position.x, self.position.y - 0.05, self.position.z)
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

    /// Adjust tilt angle within bounds
    pub fn adjust_angle(&mut self, delta: f32) {
        self.tilt_deg = (self.tilt_deg + delta).clamp(0.0, 30.0);
    }

    /// Adjust length within bounds
    pub fn adjust_length(&mut self, delta: f32) {
        let raw = self.length + delta;
        self.length = (raw * 10.0).round() / 10.0;
        self.length = self.length.clamp(0.3, 2.0);
    }

    /// Adjust width within bounds
    pub fn adjust_width(&mut self, delta: f32) {
        let raw = self.width + delta;
        self.width = (raw * 20.0).round() / 20.0;
        self.width = self.width.clamp(0.2, 1.5);
    }

    /// Adjust hole size within bounds (1mm to 20mm)
    pub fn adjust_hole_size(&mut self, delta: f32) {
        self.hole_size = (self.hole_size + delta).clamp(0.001, 0.020);
    }

    /// Adjust end width within bounds
    pub fn adjust_end_width(&mut self, delta: f32) {
        let raw = self.end_width + delta;
        // Round to nearest 0.05 to avoid float drift
        self.end_width = (raw * 20.0).round() / 20.0;
        self.end_width = self.end_width.clamp(0.15, 1.5);
    }

    /// Adjust wall height within bounds
    pub fn adjust_wall_height(&mut self, delta: f32) {
        let raw = self.wall_height + delta;
        self.wall_height = (raw * 100.0).round() / 100.0;
        self.wall_height = self.wall_height.clamp(0.02, 0.5);
    }

    /// Get width at a position along the deck (0.0 = inlet, 1.0 = outlet)
    pub fn width_at(&self, t: f32) -> f32 {
        self.width + (self.end_width - self.width) * t.clamp(0.0, 1.0)
    }

    /// Maximum width (for grid sizing)
    pub fn max_width(&self) -> f32 {
        self.width.max(self.end_width)
    }
}

/// Piece type enum for selection
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieceType {
    Gutter,
    Sluice,
    Emitter,
    ShakerDeck,
}

/// Selection state
#[derive(Clone, Copy, Debug, Default)]
pub enum Selection {
    #[default]
    None,
    Gutter(usize),
    Sluice(usize),
    Emitter(usize),
    ShakerDeck(usize),
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
    PlaceShakerDeck,
}

/// Editor layout containing all pieces
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EditorLayout {
    pub gutters: Vec<GutterPiece>,
    pub sluices: Vec<SluicePiece>,
    pub emitters: Vec<EmitterPiece>,
    #[serde(default)]
    pub shaker_decks: Vec<ShakerDeckPiece>,
    next_id: u32,
}

impl EditorLayout {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a pre-connected layout: Shaker Deck → Funnel Gutter → Sluice
    /// With emitter positioned above the shaker deck inlet
    pub fn new_connected() -> Self {
        let mut layout = Self::default();

        // === SHAKER DECK (top) ===
        // Material enters at inlet (high end), oversize exits at outlet (low end)
        // Fines fall through the perforated deck
        let shaker_id = layout.next_id();
        let mut shaker = ShakerDeckPiece::new(shaker_id);
        shaker.position = Vec3::new(0.0, 1.1, 0.0);
        shaker.rotation = Rotation::R0;
        shaker.tilt_deg = 8.0; // Moderate tilt for material flow
        shaker.length = 0.8;
        shaker.width = 0.5;
        shaker.end_width = 0.35; // Slight funnel toward outlet
                                 // Use larger holes/bars that can be resolved at sim cell size (25mm)
                                 // More bars than holes for better water interaction
        shaker.hole_size = 0.025; // 25mm holes (1 cell)
        shaker.bar_thickness = 0.05; // 50mm bars (2 cells) - 67% solid

        // === FUNNEL GUTTER (below shaker deck) ===
        // Catches fines falling through the shaker deck
        // Wide inlet to catch full width, narrow outlet feeds sluice
        let gutter_id = layout.next_id();
        let mut gutter = GutterPiece::new(gutter_id);
        gutter.rotation = Rotation::R0;
        gutter.length = 0.7;
        gutter.width = 0.6; // Wide inlet to catch fines
        gutter.end_width = 0.25; // Narrow outlet for sluice
        gutter.angle_deg = 12.0; // Steep enough for good flow

        // Position funnel gutter below shaker deck underside
        let shaker_underside = shaker.underside_position();
        let gutter_half_drop = gutter.height_drop() / 2.0;
        gutter.position = Vec3::new(
            shaker_underside.x,
            shaker_underside.y - 0.08 - gutter_half_drop, // Gap below deck
            shaker_underside.z,
        );

        // === SLUICE (bottom) ===
        // Recovery sluice catches gold from funnel gutter
        let sluice_id = layout.next_id();
        let mut sluice = SluicePiece::new(sluice_id);
        sluice.rotation = Rotation::R0;
        sluice.length = 1.2;
        sluice.width = 0.25; // Match funnel gutter outlet width

        // Position sluice so inlet aligns with funnel gutter outlet
        let gutter_outlet = gutter.outlet_position();
        let sluice_half_drop = sluice.height_drop() / 2.0;
        sluice.position = Vec3::new(
            gutter_outlet.x + sluice.length / 2.0,
            gutter_outlet.y - sluice_half_drop,
            gutter_outlet.z,
        );

        // === EMITTER (above shaker deck) ===
        let emitter_id = layout.next_id();
        let mut emitter = EmitterPiece::new(emitter_id);
        let shaker_inlet = shaker.inlet_position();
        emitter.position = Vec3::new(
            shaker_inlet.x,
            shaker_inlet.y + 0.18, // Above shaker inlet
            shaker_inlet.z,
        );
        emitter.rotation = Rotation::R0;
        emitter.rate = 1500.0; // Good flow rate for shaker

        layout.shaker_decks.push(shaker);
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

    /// Add a new shaker deck at position
    pub fn add_shaker_deck(&mut self, position: Vec3) -> usize {
        let id = self.next_id();
        let mut deck = ShakerDeckPiece::new(id);
        deck.position = position;
        self.shaker_decks.push(deck);
        self.shaker_decks.len() - 1
    }

    /// Remove a shaker deck by index
    pub fn remove_shaker_deck(&mut self, index: usize) {
        if index < self.shaker_decks.len() {
            self.shaker_decks.remove(index);
        }
    }

    /// Get total piece count
    pub fn piece_count(&self) -> usize {
        self.gutters.len() + self.sluices.len() + self.emitters.len() + self.shaker_decks.len()
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

/// Scenario config for editor + headless tests (layout + optional sim/test metadata).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScenarioSimConfig {
    #[serde(default)]
    pub cell_size: Option<f32>,
    #[serde(default)]
    pub pressure_iters: Option<usize>,
    #[serde(default)]
    pub substeps: Option<u32>,
    #[serde(default)]
    pub gravity: Option<f32>,
    #[serde(default)]
    pub max_particles: Option<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FlowDirectionCheck {
    #[serde(default)]
    pub axis: Option<String>,
    #[serde(default)]
    pub min_avg_vel: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PoolEquilibriumCheck {
    #[serde(default)]
    pub max_rms_vel: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WallContainmentCheck {
    #[serde(default)]
    pub max_escape: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScenarioSpawnConfig {
    pub count: usize,
    #[serde(with = "vec3_serde")]
    pub min: Vec3,
    #[serde(with = "vec3_serde")]
    pub max: Vec3,
    #[serde(with = "vec3_serde")]
    pub velocity: Vec3,
    #[serde(default)]
    pub use_gold: bool,
    #[serde(default)]
    pub seed: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScenarioChecks {
    #[serde(default)]
    pub flow_direction: Option<FlowDirectionCheck>,
    #[serde(default)]
    pub pool_equilibrium: Option<PoolEquilibriumCheck>,
    #[serde(default)]
    pub wall_containment: Option<WallContainmentCheck>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScenarioTestConfig {
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub duration_s: Option<f32>,
    #[serde(default)]
    pub emit_for_s: Option<f32>,
    #[serde(default)]
    pub spawn: Option<ScenarioSpawnConfig>,
    #[serde(default)]
    pub checks: ScenarioChecks,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScenarioConfig {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub sim: Option<ScenarioSimConfig>,
    #[serde(default)]
    pub test: Option<ScenarioTestConfig>,
    #[serde(flatten)]
    pub layout: EditorLayout,
}

impl ScenarioConfig {
    pub fn from_layout(layout: EditorLayout) -> Self {
        Self {
            layout,
            ..Default::default()
        }
    }

    pub fn save_json(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_json(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let config = serde_json::from_str(&json)?;
        Ok(config)
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
