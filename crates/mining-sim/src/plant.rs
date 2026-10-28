//! First saved processing circuit: existing finite hopper, actual dry screen,
//! receiving gutter and wet sluice. Dense-bed/wet-deck hydraulics remain open.
use crate::{
    screen::Screen,
    sluice::{Grain, SCREEN_FEED_BATCH_MAX},
    Material, Sluice, Vec3,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Port {
    ScreenUndersize,
    SluiceFeed,
    ScreenOversize,
    RejectBin,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Connection {
    pub source_module: u64,
    pub source: Port,
    pub destination_module: u64,
    pub destination: Port,
}
fn routes() -> Vec<Connection> {
    vec![
        Connection {
            source_module: 1,
            source: Port::ScreenUndersize,
            destination_module: 0,
            destination: Port::SluiceFeed,
        },
        Connection {
            source_module: 1,
            source: Port::ScreenOversize,
            destination_module: 2,
            destination: Port::RejectBin,
        },
    ]
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plant {
    pub model_version: u32,
    pub screen: Screen,
    /// Stable module IDs: sluice 0, screen 1, reject destination 2. The first
    /// implemented circuit admits precisely these exclusive typed routes.
    pub connections: Vec<Connection>,
    /// Emitted but not yet accepted by the screen. This is an owned donor,
    /// including exact source grading/numerical state across any refusal.
    pub pending_feed: Vec<Grain>,
    pub pending_moisture_m3: f64,
}
impl Plant {
    pub fn new(s: &Sluice) -> Result<Self, String> {
        Self::for_placement(
            s,
            s.placement
                .ok_or("Place the sluice before fitting its screen")?,
        )
    }
    fn for_placement(s: &Sluice, p: crate::Placement) -> Result<Self, String> {
        if s.profile != crate::sluice::HydraulicProfile::OriginalDemo || !s.granular_packets {
            return Err("This screen connects to the current granular sluice profile".into());
        }
        let screen = Screen::new(p.inlet - p.forward() * 0.70 + Vec3::Y * 0.60, p.yaw);
        if Vec3::new(
            screen.origin.x - crate::sluice::HOPPER_X,
            0.,
            screen.origin.z - crate::sluice::HOPPER_Z,
        )
        .length()
            > 0.75
        {
            return Err("Screen inlet must connect beneath the fixed hopper chute".into());
        }
        Ok(Self {
            model_version: 1,
            screen,
            connections: routes(),
            pending_feed: Vec::new(),
            pending_moisture_m3: 0.,
        })
    }
    pub fn relocated(&self, s: &Sluice, p: crate::Placement) -> Result<Self, String> {
        if self.screen.motor_on || self.material().mass_kg() > 0. || self.water_m3() > 0. {
            return Err("Stop and empty the connected screen before moving it".into());
        }
        let mut next = Self::for_placement(s, p)?;
        let origin = next.screen.origin;
        next.model_version = self.model_version;
        next.screen = self.screen.clone();
        next.screen.origin = origin;
        next.screen.yaw = p.yaw;
        Ok(next)
    }
    pub fn set_bulk_screen(
        &mut self,
        config: Option<crate::screen::BulkDeck>,
    ) -> Result<(), String> {
        if !self.valid() || !self.pending_feed.is_empty() || self.pending_moisture_m3 != 0. {
            return Err(
                "Empty the complete feeder circuit before changing its screen model".into(),
            );
        }
        self.screen.set_bulk_deck(config)?;
        // Older readers reject this model instead of silently reverting its law.
        self.model_version = self.screen.processing_model_version();
        Ok(())
    }
    pub fn valid(&self) -> bool {
        self.model_version == self.screen.processing_model_version()
            && self.connections == routes()
            && self.screen.valid()
            && self.pending_feed.len() <= SCREEN_FEED_BATCH_MAX
            && self.pending_feed.iter().all(crate::tailings::valid_grain)
            && self.pending_moisture_m3.is_finite()
            && self.pending_moisture_m3 >= 0.
    }
    pub fn connected_to(&self, s: &Sluice) -> bool {
        let Ok(expected) = Self::new(s) else {
            return false;
        };
        self.valid()
            && self.screen.origin.distance(expected.screen.origin) < 1e-5
            && (self.screen.yaw - expected.screen.yaw).abs() < 1e-6
            && (self.screen.config.length_m - 0.8).abs() < 1e-12
            && (self.screen.config.width_m - 0.38).abs() < 1e-12
    }
    pub fn material(&self) -> Material {
        let mut m = self.screen.material();
        for g in &self.pending_feed {
            m.add(g.material);
        }
        m
    }
    pub fn water_m3(&self) -> f64 {
        self.material().pore_water_m3 + self.screen.free_water_m3 + self.pending_moisture_m3
    }
    /// Reserve the complete receiver before relinquishing any output. Stored
    /// pending grain state is never collapsed into bulk and resampled here.
    fn admit_underflow(&mut self, s: &mut Sluice) -> Result<(), String> {
        // The unchanged granular solver advances two fixed ticks together.
        // Admit only at the beginning of that interval, never age a product
        // through time it still spent on the screen.
        if s.fluid_clock > 1e-6 {
            return Ok(());
        }
        if self.screen.undersize.is_empty() {
            return Ok(());
        }
        let moisture: f64 = self
            .screen
            .undersize
            .iter()
            .map(|g| g.material.pore_water_m3)
            .sum();
        let room = s
            .profile
            .capacity()
            .saturating_sub(s.water.len() + s.outflow.len()) as f64
            * s.water_particle_m3()
            - s.water_credit_m3
            - s.feed_water_credit_m3;
        if moisture > room.max(0.) {
            return Ok(());
        }
        let mut incoming = Vec::new();
        incoming
            .try_reserve(self.screen.undersize.len())
            .map_err(|_| "Cannot reserve sluice output; screen retains grains")?;
        for g in &self.screen.undersize {
            let mut moving = g.clone();
            moving.resting = false;
            let mut grid = crate::tailings::grid_grain(s, &moving)?;
            // A disconnected outlet holds its products. It never teleports
            // them into the receiver or silently turns a miss into recovery.
            let radius = grid.diameter_m * 0.5;
            // Match the unchanged original solver's vertical domain (geometry.rs).
            // A high/ricocheting arrival stays in its product owner rather than
            // being admitted above the roof and lost by the domain filter.
            let roof = (s.floor_cells() + 22) as f32 * s.profile.cell();
            if grid.position.y <= radius
                || grid.position.y + radius >= roof
                || grid.position.x < s.profile.cell()
                || grid.position.x > s.profile.length()
                || grid.position.z < s.profile.cell()
                || grid.position.z > s.profile.cell() + s.profile.width()
            {
                return Ok(());
            }
            grid.material.pore_water_m3 = 0.;
            incoming.push(grid);
        }
        s.grains
            .try_reserve(incoming.len())
            .map_err(|_| "Cannot reserve sluice grains; screen retains products")?;
        let delivered = self.screen.take_undersize();
        s.processed_m3 += delivered.iter().map(|g| g.material.volume()).sum::<f64>();
        s.water_used_m3 += moisture;
        s.feed_water_credit_m3 += moisture;
        s.grains.extend(incoming);
        Ok(())
    }
    pub fn step(&mut self, s: &mut Sluice, dt: f32, water_available: bool) -> Result<(), String> {
        if !self.connected_to(s) {
            return Err("Invalid saved plant; material retained".into());
        }
        self.admit_underflow(s)?;
        self.screen.step(dt)?;
        // Newly discharged bodies have already consumed this screen tick.
        // Transfer them at the next tick's start so the sluice cannot advance
        // them through the same physical dt a second time.
        if self.pending_feed.is_empty()
            && self.pending_moisture_m3 == 0.
            && self.screen.motor_on
            && s.feed_on
            && s.pump_on
            && water_available
        {
            // Refusing at a numerical capacity is not a physical clogging law.
            self.screen.try_reserve_feed(SCREEN_FEED_BATCH_MAX)?;
            let batch = s.take_screen_feed(dt, water_available)?;
            self.pending_feed = batch.grains;
            self.pending_moisture_m3 = batch.moisture_m3;
        }
        if !self.pending_feed.is_empty() || self.pending_moisture_m3 > 0. {
            self.screen.try_reserve_feed(self.pending_feed.len())?;
            let mut world = Vec::new();
            world
                .try_reserve(self.pending_feed.len())
                .map_err(|_| "Cannot reserve feeder transfer; donor retained")?;
            for g in &self.pending_feed {
                let across = g.position.z - s.profile.width() * 0.5 - s.profile.cell();
                let mut out = crate::tailings::world_grain(s, g)?;
                // These are newly emitted feeder grains, not relocated stored
                // particles. Birth follows the actual upstream chute footprint.
                out.position = self
                    .screen
                    .to_world(glam::DVec3::new(
                        0.035 + (g.position.x as f64 - 0.10),
                        0.055,
                        across as f64,
                    ))
                    .as_vec3();
                out.position += self.screen.offset_at(self.screen.time).0.as_vec3();
                world.push(out);
            }
            self.screen.admit(&world, self.pending_moisture_m3)?;
            self.pending_feed.clear();
            self.pending_moisture_m3 = 0.;
        }
        Ok(())
    }
    /// Explicit operator handling into the existing bulk hand inventory. Its
    /// grading/history limitation is inherited, not a screen recovery bonus.
    pub fn clear_rejects(&mut self) -> Result<Material, String> {
        let mut m = Material::default();
        for g in self.screen.cleanout()? {
            m.add(g.material);
        }
        for g in self.pending_feed.drain(..) {
            m.add(g.material);
        }
        m.pore_water_m3 += self.screen.take_free_water() + self.pending_moisture_m3;
        self.pending_moisture_m3 = 0.;
        Ok(m)
    }
    pub fn release_spills(&mut self, t: &mut crate::tailings::Tailings) -> Result<(), String> {
        if !self.screen.spills.iter().all(crate::tailings::valid_grain) {
            return Err("Invalid screen spill; donor retained".into());
        }
        t.prepare_batch(self.screen.spills.len())?;
        for grain in self.screen.take_spills() {
            t.accept(&grain).expect("validated reserved spill transfer");
        }
        Ok(())
    }
}
