use crate::{
    machine::MachineInput,
    site::SiteSnapshot,
    sluice::{HOPPER_X, HOPPER_Z},
    Excavator, Material, Placement, Site, Sluice, Vec3,
};
use serde::{Deserialize, Serialize};

pub const FIXED_DT: f32 = 1.0 / 60.0;
pub const GOLD_TARGET_G: f64 = 5.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlMode {
    OnFoot,
    Excavator,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Player {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub mode: ControlMode,
    pub third_person: bool,
    pub hand: Material,
    pub tool_swing: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FallingLoad {
    pub position: Vec3,
    pub velocity: Vec3,
    pub material: Material,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rigid: Option<crate::payload::RigidState>,
}
#[derive(Clone, Copy, Debug)]
pub enum Action {
    EnterExit,
    Camera,
    TogglePump,
    ToggleFeed,
    Engine,
    Cleanup,
    Place(Placement),
    Dig(Vec3),
    DumpHand,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Controls {
    pub walk: f32,
    pub strafe: f32,
    pub sprint: bool,
    pub machine: MachineInput,
}

#[derive(Clone)]
pub struct Session {
    pub site: Site,
    pub machine: Excavator,
    pub sluice: Sluice,
    pub player: Player,
    pub falling: Vec<FallingLoad>,
    pub sim_time: f64,
    pub paused: bool,
    pub message: String,
    pub initial_gold_g: f64,
    pub delivered_m3: f64,
    pub water_balance: crate::water_balance::WaterBalance,
    pub material_motion: Option<crate::payload::MaterialMotion>,
    pub active_soil: Option<crate::soil::ActiveSoil>,
}
#[derive(Serialize, Deserialize)]
struct Save {
    version: u32,
    site: SiteSnapshot,
    machine: Excavator,
    sluice: Sluice,
    player: Player,
    falling: Vec<FallingLoad>,
    sim_time: f64,
    initial_gold_g: f64,
    delivered_m3: f64,
    #[serde(default)]
    water_balance: Option<crate::water_balance::WaterBalance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    material_motion: Option<crate::payload::MaterialMotion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    active_soil: Option<crate::soil::ActiveSoil>,
}
impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}
impl Session {
    pub fn new() -> Self {
        let site = Site::new();
        let machine = Excavator::new(&site);
        let initial_gold_g = site.remaining_gold();
        let water_balance = crate::water_balance::WaterBalance::new(
            site.water_volume_m3() + site.soil_water.stored_m3(),
            0.,
            false,
        );
        let player = Player {
            position: Vec3::new(21.6, site.height(21.6, 17.0), 17.0),
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: -0.12,
            mode: ControlMode::OnFoot,
            third_person: false,
            hand: Material::default(),
            tool_swing: 0.0,
        };
        Self {
            site,
            machine,
            sluice: Sluice::default(),
            player,
            falling: Vec::new(),
            sim_time: 0.0,
            paused: false,
            message: "Welcome to First Bend. Enter the excavator with E. Controls: F1.".into(),
            initial_gold_g,
            delivered_m3: 0.0,
            water_balance,
            material_motion: None,
            active_soil: None,
        }
    }
    pub fn near_plant(&self) -> bool {
        self.player
            .position
            .distance(Sluice::hopper_position(&self.site))
            < 5.0
    }
    pub fn look_direction(&self) -> Vec3 {
        Vec3::new(
            self.player.yaw.sin() * self.player.pitch.cos(),
            self.player.pitch.sin(),
            self.player.yaw.cos() * self.player.pitch.cos(),
        )
    }
    /// Preview and commit share all placement checks, including player reach
    /// and the requirement to drain/clean an existing active installation.
    pub fn validate_placement(&self, placement: Placement) -> Result<(), String> {
        if self.paused {
            return Err("Resume the shift first".into());
        }
        if self.player.mode != ControlMode::OnFoot {
            return Err("Exit the excavator to place equipment".into());
        }
        if self.player.position.distance(placement.inlet) > 9.0 {
            return Err("Move closer to the placement".into());
        }
        self.sluice
            .validate_relocation(placement, &self.site, self.machine.position)
    }
    pub fn on_action(&mut self, a: Action) -> Result<(), String> {
        if self.paused {
            return Err("Resume the shift first".into());
        }
        let result = match a {
            Action::Camera => {
                self.player.third_person = !self.player.third_person;
                Ok(())
            }
            Action::EnterExit => match self.player.mode {
                ControlMode::OnFoot => {
                    if self.player.position.distance(self.machine.position) > 3.5 {
                        Err("Walk up to the excavator cab".into())
                    } else {
                        self.player.mode = ControlMode::Excavator;
                        self.player.yaw = self.machine.heading + self.machine.swing;
                        self.player.pitch = -0.18;
                        Ok(())
                    }
                }
                ControlMode::Excavator => {
                    let right =
                        Vec3::new(self.machine.heading.cos(), 0.0, -self.machine.heading.sin());
                    let mut found = None;
                    for offset in [
                        right * 2.5,
                        -right * 2.5,
                        -self.machine.forward() * 3.0,
                        self.machine.forward() * 3.0,
                    ] {
                        let p = self.machine.position + offset;
                        if self.walkable(p) {
                            found = Some(p);
                            break;
                        }
                    }
                    if let Some(p) = found {
                        self.player.mode = ControlMode::OnFoot;
                        self.player.position = Vec3::new(p.x, self.site.height(p.x, p.z), p.z);
                        Ok(())
                    } else {
                        Err("No safe ground beside the excavator".into())
                    }
                }
            },
            Action::Engine => {
                if self.player.mode == ControlMode::Excavator {
                    self.machine.engine_on = !self.machine.engine_on;
                    Ok(())
                } else {
                    Err("Enter the excavator to operate its engine".into())
                }
            }
            Action::Place(p) => self.validate_placement(p).and_then(|()| {
                self.sluice.relocate(p, &self.site, self.machine.position)?;
                self.message = "Sluice installed. P starts the pump; O starts the feed.".into();
                Ok(())
            }),
            Action::TogglePump | Action::ToggleFeed | Action::Cleanup => {
                if !self.near_plant() || self.player.mode != ControlMode::OnFoot {
                    Err("Walk to the processing plant".into())
                } else if self.sluice.placement.is_none() {
                    Err("Place the sluice first: B".into())
                } else {
                    match a {
                        Action::TogglePump => {
                            self.sluice.pump_on = !self.sluice.pump_on;
                            Ok(())
                        }
                        Action::ToggleFeed => {
                            self.sluice.feed_on = !self.sluice.feed_on;
                            Ok(())
                        }
                        _ => self.sluice.cleanup().map(|m| {
                            self.message = format!(
                                "Cleanup {}: {:.2} g of gold recovered",
                                self.sluice.cleanups, m.gold_g
                            );
                        }),
                    }
                }
            }
            Action::Dig(p) => {
                if self.player.mode != ControlMode::OnFoot {
                    Err("Use the machine's bucket controls".into())
                } else if !p.is_finite() || self.player.position.distance(p) > 2.8 {
                    Err("Move closer to the ground you want to dig".into())
                } else {
                    // Carrying is unbounded for now; the shovel footprint,
                    // cut depth and finite terrain still bound each stroke.
                    let m = self.site.dig(p, 0.22, 0.12, f64::MAX);
                    if m.is_empty() {
                        Err("Nothing left to dig here.".into())
                    } else {
                        self.player.hand.add(m);
                        self.player.tool_swing = 1.0;
                        Ok(())
                    }
                }
            }
            Action::DumpHand => {
                if self.player.mode != ControlMode::OnFoot || self.player.hand.is_empty() {
                    Err("Your hand bucket is empty".into())
                } else {
                    let forward = Vec3::new(self.player.yaw.sin(), 0.0, self.player.yaw.cos());
                    let position = self.player.position + Vec3::Y * 1.85 + forward * 0.60;
                    let m = self.player.hand.take_fraction(1.0);
                    self.falling.push(FallingLoad {
                        position,
                        velocity: forward * 0.7,
                        material: m,
                        rigid: None,
                    });
                    Ok(())
                }
            }
        };
        if let Err(ref message) = result {
            self.message = message.clone();
        } else {
            match a {
                Action::EnterExit => {
                    self.message = if self.player.mode == ControlMode::Excavator {
                        if self.active_soil.is_some() {
                            "Cab controls: R/F boom, T/G stick, Y/H curl. Move the bucket into the finite bank.".into()
                        } else if self.machine.articulation.is_some() {
                            "Cab controls: R/F boom, T/G stick, Y/H curl. Mechanism validation: digging and dumping unavailable.".into()
                        } else {
                            "Cab controls: R/F boom, T/G stick, Y/H curl. Hold Space to cut; J pours the bucket.".into()
                        }
                    } else {
                        "Back on foot. Walk to the hopper to operate the plant.".into()
                    }
                }
                Action::Engine => {
                    self.message = if self.machine.engine_on {
                        "Engine started".into()
                    } else {
                        "Engine stopped".into()
                    }
                }
                Action::TogglePump => {
                    self.message = if self.sluice.pump_on {
                        "Pump on. Let the sluice fill, then open the feed with O.".into()
                    } else {
                        "Pump off. Stop feed and let moving material settle before cleanup.".into()
                    }
                }
                Action::ToggleFeed => {
                    self.message = if self.sluice.feed_on {
                        "Feed open. Your hopper material is entering the sluice.".into()
                    } else {
                        "Feed stopped. Leave water running to flush the remaining gangue.".into()
                    }
                }
                _ => {}
            }
        }
        result
    }
    fn walkable(&self, p: Vec3) -> bool {
        if !Site::bounds(p, 0.6) || self.site.water_depth(p.x, p.z) > 0.5 {
            return false;
        }
        let m = self.machine.position;
        if Vec3::new(p.x - m.x, 0.0, p.z - m.z).length() < 1.9 {
            return false;
        }
        let hopper = Sluice::hopper_position(&self.site);
        if Vec3::new(p.x - hopper.x, 0.0, p.z - hopper.z).length() < 1.0 {
            return false;
        }
        if let Some(s) = self.sluice.placement {
            let d = p - s.inlet;
            let x = d.dot(s.forward());
            let z = d.dot(s.right()).abs();
            if (-0.2..self.sluice.profile.length() + 0.2).contains(&x)
                && z < self.sluice.profile.width() * 0.5 + 0.24
            {
                return false;
            }
        }
        true
    }
    pub fn step(&mut self, input: Controls) {
        if self.paused {
            return;
        }
        let dt = FIXED_DT;
        self.sim_time += dt as f64;
        self.player.tool_swing = (self.player.tool_swing - dt * 2.8).max(0.0);
        let hopper = Sluice::hopper_position(&self.site);
        let mut obstacles = vec![crate::tracks::Obstacle {
            center: hopper + Vec3::Y * 0.35,
            rotation: glam::Quat::IDENTITY,
            half_extents: Vec3::new(0.6, 0.25, 0.6),
        }];
        for side in [-1., 1.] {
            for axis in [Vec3::X, Vec3::Z] {
                obstacles.push(crate::tracks::Obstacle {
                    center: hopper + Vec3::Y * 1.20 + axis * (side * 0.55),
                    rotation: glam::Quat::IDENTITY,
                    half_extents: if axis == Vec3::X {
                        Vec3::new(0.025, 0.35, 0.55)
                    } else {
                        Vec3::new(0.55, 0.35, 0.025)
                    },
                });
            }
        }
        if let Some(p) = self.sluice.placement {
            let rotation = glam::Quat::from_rotation_y(p.yaw)
                * glam::Quat::from_rotation_x(p.slope_deg.to_radians());
            obstacles.push(crate::tracks::Obstacle {
                center: p.to_world(Vec3::new(self.sluice.profile.length() * 0.5, -0.025, 0.)),
                rotation,
                half_extents: Vec3::new(
                    self.sluice.profile.width() * 0.5 + 0.04,
                    0.025,
                    self.sluice.profile.length() * 0.5,
                ),
            });
        }
        if self.player.mode == ControlMode::Excavator {
            // Equipment, rather than camera focus, owns the running simulation.
            let old_yaw = self.machine.heading + self.machine.swing;
            // Reject blocked travel before it reaches the tool sweep. Reverting
            // the chassis after cutting could mine at a rejected bucket pose.
            let heading = self.machine.heading + input.machine.steer.clamp(-1., 1.) * dt * 0.45;
            let p = self.machine.position
                + Vec3::new(heading.sin(), 0., heading.cos())
                    * input.machine.drive.clamp(-1., 1.)
                    * dt
                    * 1.6;
            let hopper = Sluice::hopper_position(&self.site);
            let blocks_hopper = Vec3::new(p.x - hopper.x, 0.0, p.z - hopper.z).length() < 2.2;
            let blocks_sluice = self.sluice.placement.is_some_and(|placement| {
                let d = p - placement.inlet;
                let x = d.dot(placement.forward());
                let z = d.dot(placement.right()).abs();
                (-1.8..self.sluice.profile.length() + 1.8).contains(&x) && z < 1.7
            });
            let mut machine_input = input.machine;
            if blocks_hopper || blocks_sluice {
                machine_input.drive = 0.;
            }
            if let Some((position, velocity, material)) = self.machine.step_with_soil(
                machine_input,
                &mut self.site,
                dt,
                &obstacles,
                self.active_soil.as_mut(),
            ) {
                self.falling.push(FallingLoad {
                    position,
                    velocity,
                    material,
                    rigid: None,
                });
            }
            self.player.yaw += self.machine.heading + self.machine.swing - old_yaw;
            self.player.position = self.machine.position + Vec3::Y * 1.9;
        } else {
            self.machine.step_with_soil(
                MachineInput::default(),
                &mut self.site,
                dt,
                &obstacles,
                self.active_soil.as_mut(),
            );
            let forward = Vec3::new(self.player.yaw.sin(), 0.0, self.player.yaw.cos());
            // Bevy looks along local -Z: screen-right is forward × up.
            let right = forward.cross(Vec3::Y);
            let speed = if input.sprint { 4.2 } else { 2.4 };
            let delta =
                (forward * input.walk + right * input.strafe).clamp_length_max(1.0) * speed * dt;
            let current = self.player.position;
            // Resolve each horizontal axis independently so a blocked diagonal
            // slides along equipment instead of cancelling all walking.
            let mut position = current;
            for offset in [Vec3::X * delta.x, Vec3::Z * delta.z] {
                let next = position + offset;
                if self.walkable(next)
                    && (self.site.height(next.x, next.z) - position.y).abs() < 0.45
                {
                    position = Vec3::new(next.x, self.site.height(next.x, next.z), next.z);
                }
            }
            self.player.position = position;
        }
        let top = Sluice::hopper_top(&self.site);
        let mut impacts = Vec::new();
        if self.active_soil.is_some() {
            if let Some(error) = self
                .machine
                .articulation
                .as_ref()
                .and_then(|a| a.feedback.failure.as_ref())
            {
                self.paused = true;
                self.message = format!("Active soil stopped: {error}");
                return;
            }
        }
        for (i, load) in self.falling.iter_mut().enumerate() {
            let before = load.position;
            match load.advance_rigid(dt) {
                Ok(true) => {}
                Ok(false) => {
                    load.velocity.y -= 9.81 * dt;
                    load.position += load.velocity * dt;
                }
                Err(error) => {
                    self.paused = true;
                    self.message = format!("Material motion paused: {error}");
                    return;
                }
            }
            let p = load.position;
            let over_hopper = (p.x - HOPPER_X).abs() < 0.85 && (p.z - HOPPER_Z).abs() < 0.85;
            if over_hopper && before.y >= top - 0.03 && p.y <= top {
                impacts.push((i, true));
            } else if p.y <= self.site.height(p.x, p.z) || !Site::bounds(p, 0.2) {
                impacts.push((i, false));
            }
        }
        for (i, hopper) in impacts.into_iter().rev() {
            let load = self.falling.swap_remove(i);
            if let Some(parcel) = load.rigid_parcel() {
                let motion = self.material_motion.get_or_insert_with(Default::default);
                if hopper {
                    motion.hopper.receive(&parcel);
                } else {
                    motion.terrain.receive(&parcel);
                }
            }
            if hopper {
                let overflow = self.sluice.add_to_hopper(load.material);
                self.delivered_m3 += load.material.volume() - overflow.volume();
                if !overflow.is_empty() {
                    self.site.deposit(load.position + Vec3::X * 1.3, overflow);
                }
            } else {
                self.site.deposit(load.position, load.material);
            }
        }
        crate::outflow::step(&mut self.sluice, &mut self.site, dt);
        let requested = if self.sluice.pump_on
            && self.sluice.placement.is_some()
            && self.sluice.intake_has_room()
        {
            let room = self
                .sluice
                .profile
                .capacity()
                .saturating_sub(self.sluice.water.len() + self.sluice.outflow.len())
                as f64
                * self.sluice.profile.particle_m3()
                - self.sluice.water_credit_m3
                - self.sluice.feed_water_credit_m3;
            (self.sluice.flow_m3s * dt).min(room.max(0.0) as f32)
        } else {
            0.0
        };
        let actual = if requested > 0.0 {
            self.site.withdraw_intake(
                Vec3::new(Site::river_center(HOPPER_Z), 0.0, HOPPER_Z),
                requested,
            )
        } else {
            crate::intake::IntakePacket::default()
        };
        let expected = self.sluice.flow_m3s * dt;
        let fraction = if expected > 0.0 {
            actual.water_m3 / expected as f64
        } else {
            0.0
        };
        self.sluice.water_used_m3 += actual.water_m3;
        self.sluice.inlet.last_water_m3 = actual.water_m3;
        self.sluice.accept_intake(actual);
        self.sluice.feed(dt, fraction > 0.01);
    }

    pub fn accounted_gold(&self) -> f64 {
        self.site.remaining_gold()
            + self.site.hydraulic.exported_gold_g
            + self.machine.bucket.gold_g
            + self.player.hand.gold_g
            + self.sluice.active_material().gold_g
            + self.sluice.tailings.gold_g
            + self.sluice.concentrate.gold_g
            + self.falling.iter().map(|l| l.material.gold_g).sum::<f64>()
            + self
                .active_soil
                .as_ref()
                .map_or(0., |s| s.material().gold_g)
    }
    pub fn accounted_bulk_material_m3(&self) -> f64 {
        self.accounted_material().volume()
    }
    pub fn accounted_dry_mass_kg(&self) -> f64 {
        self.accounted_material().mass_kg()
    }
    pub fn accounted_material(&self) -> Material {
        let mut total = self.site.stored_material();
        for m in [
            crate::intake::Suspension {
                solid_m3: self.site.hydraulic.exported_material_m3,
                gold_g: self.site.hydraulic.exported_gold_g,
            }
            .material(),
            self.machine.bucket,
            self.player.hand,
            self.sluice.active_material(),
            self.sluice.tailings,
            self.sluice.concentrate,
        ] {
            total.add(m);
        }
        for load in &self.falling {
            total.add(load.material);
        }
        if let Some(soil) = &self.active_soil {
            total.add(soil.material());
        }
        total
    }
    /// Finite terrain-to-moving-soil activation. Cohesion and wet contacts are
    /// intentionally not synthesized by the first frictional model.
    /// Declared finite initial condition for interactive drained-soil checks.
    /// The pose is prepared before the clock starts; subsequent motion uses
    /// ordinary actuator commands and actual moving soil/steel contacts.
    pub fn prepare_soil_trial(&mut self) -> Result<(), String> {
        if self.sim_time != 0. || self.active_soil.is_some() || !self.machine.bucket.is_empty() {
            return Err("Soil trial needs a fresh empty machine".into());
        }
        let clearance = |m: &Excavator| {
            let a = m.arm();
            let q = m.bucket_rotation();
            m.bucket_model()
                .geometry()
                .parts
                .iter()
                .flat_map(|p| p.vertices.iter())
                .map(|v| a.wrist + q * (*v))
                .map(|p| p.y - self.site.height(p.x, p.z))
                .fold(f32::INFINITY, f32::min)
        };
        let mut m = self.machine.clone();
        if m.tracks.carries_motion()
            || m.articulation
                .as_ref()
                .is_some_and(|a| a.velocity.iter().any(|v| *v != 0.))
        {
            return Err("Soil trial cannot discard an initially moving machine".into());
        }
        // The CLI may already have selected articulation. Rebuild its initial
        // zero-rate chart after preparing the initial boom angle, before time 0.
        m.articulation = None;
        m.enable_articulation()?;
        let (mut lo, mut hi) = (-0.24, 1.20);
        m.boom = lo;
        if clearance(&m) >= 0.015 {
            return Err("Soil trial cannot bracket the bank".into());
        }
        m.boom = hi;
        if clearance(&m) <= 0.015 {
            return Err("Soil trial cannot clear the bank".into());
        }
        for _ in 0..24 {
            m.boom = (lo + hi) * 0.5;
            if clearance(&m) < 0.015 {
                lo = m.boom;
            } else {
                hi = m.boom;
            }
        }
        m.boom = hi;
        m.articulation = None;
        m.enable_articulation()?;
        let a = m.arm();
        // Activate the finite bank under the entire curved tool as well as
        // its entry. Leaving rigid, unactivated soil under the belly blocks
        // excavation before the lip can load the bank. All removed stock is
        // retained as real clods at its original position before time zero.
        let forward = Vec3::new(a.forward.x, 0., a.forward.z).normalize();
        let right = Vec3::new(a.right.x, 0., a.right.z).normalize();
        let origin = Vec3::new(a.wrist.x, 0., a.wrist.z);
        let low = (a.lip - origin).dot(forward) - 0.85;
        let high = (a.heel - origin).dot(forward) + 0.20;
        // Heightfield nodes influence neighbouring triangles. Selecting only
        // centres inside the requested patch leaves a rigid interpolated
        // shoulder under the side plates. Include the support stencil of the
        // footprint; all additionally activated stock becomes actual clods.
        let lateral_halo = crate::site::CELL * (right.x.abs() + right.z.abs());
        let forward_halo = crate::site::CELL * (forward.x.abs() + forward.z.abs());
        let mut nodes = Vec::new();
        for z in 1..crate::site::WIDTH - 1 {
            for x in 1..crate::site::WIDTH - 1 {
                let p = Vec3::new(
                    x as f32 * crate::site::CELL,
                    0.,
                    z as f32 * crate::site::CELL,
                ) - origin;
                if p.dot(right).abs() <= 0.45 + lateral_halo
                    && (low - forward_halo..=high + forward_halo).contains(&p.dot(forward))
                {
                    nodes.push((z * crate::site::WIDTH + x, 0.18));
                }
            }
        }
        let mut site = self.site.clone();
        let soil = crate::soil::activate(&mut site, &nodes, 2)?;
        self.machine = m;
        self.site = site;
        self.active_soil = Some(soil);
        self.message="Drained frictional soil trial: move the actual bucket with R/F, T/G, Y/H. Shell contacts retain and spill independent clods; soil strength, cohesion and wet contact are not calibrated.".into();
        Ok(())
    }
    pub fn activate_soil(
        &mut self,
        nodes: &[(usize, f32)],
        subdivisions: usize,
    ) -> Result<(), String> {
        if self.machine.articulation.is_none() {
            return Err("Moving soil requires the articulated contact model".into());
        }
        let mut site = self.site.clone();
        let mut soil = crate::soil::activate(&mut site, nodes, subdivisions)?;
        if let Some(old) = &self.active_soil {
            soil.clods.extend(old.clods.clone());
            soil = crate::soil::ActiveSoil::from_clods(soil.clods)?;
        }
        self.site = site;
        self.active_soil = Some(soil);
        Ok(())
    }
    pub fn uses_material_motion(&self) -> bool {
        self.machine
            .articulation
            .as_ref()
            .is_some_and(|a| a.payload_moments.is_some() || a.exchange.is_some())
            || self.falling.iter().any(|l| l.rigid.is_some())
            || self.material_motion.is_some()
    }
    pub fn capture_rigid_load(
        &mut self,
        index: usize,
        fraction: f64,
    ) -> Result<crate::payload::ExchangeReport, String> {
        let mut source = Some(
            self.falling
                .get(index)
                .filter(|load| load.valid_rigid_state())
                .and_then(FallingLoad::rigid_parcel)
                .ok_or("No finite rigid load")?,
        );
        let report =
            crate::articulation::capture_payload(&mut self.machine, &mut source, fraction)?;
        if let Some(parcel) = source {
            self.falling[index] = parcel.into_falling();
        } else {
            self.falling.remove(index);
        }
        Ok(report)
    }
    pub fn release_rigid_payload(
        &mut self,
        fraction: f64,
    ) -> Result<crate::payload::ExchangeReport, String> {
        if self.falling.len() >= 2048 {
            return Err("No room for another finite material parcel".into());
        }
        let parcel = crate::articulation::release_payload(&mut self.machine, fraction)?;
        let report = self
            .machine
            .articulation
            .as_ref()
            .unwrap()
            .exchange
            .as_ref()
            .unwrap()
            .last
            .clone();
        self.falling.push(parcel.into_falling());
        Ok(report)
    }
    pub fn save_json(&self) -> Result<String, String> {
        let mut site = self.site.snapshot();
        site.normalize_bed_age();
        serde_json::to_string(&Save {
            // v8 preserves spatial payload/rigid parcels and boundary receipts.
            version: if self.machine.bucket_model() == crate::bucket::Model::MountedShellV2 {
                11
            } else if self
                .machine
                .articulation
                .as_ref()
                .is_some_and(|a| a.bucket_model() == crate::bucket::Model::ShellV1)
            {
                10
            } else if self.active_soil.is_some() {
                9
            } else if self.uses_material_motion() {
                8
            } else if self.machine.articulation.is_some() {
                7
            } else {
                6
            },
            site,
            machine: self.machine.clone(),
            sluice: self.sluice.clone(),
            player: self.player.clone(),
            falling: self.falling.clone(),
            sim_time: self.sim_time,
            initial_gold_g: self.initial_gold_g,
            delivered_m3: self.delivered_m3,
            water_balance: Some(self.water_balance.clone()),
            material_motion: self.material_motion.clone(),
            active_soil: self.active_soil.clone(),
        })
        .map_err(|e| e.to_string())
    }
    pub fn load_json(json: &str) -> Result<Self, String> {
        if json.len() > 128_000_000 {
            return Err("Save file is too large".into());
        }
        let s: Save = serde_json::from_str(json).map_err(|e| e.to_string())?;
        if !matches!(s.version, 2..=11) {
            return Err("Unsupported save version".into());
        }
        if !matches!(
            (s.version, s.site.bed_age_unit()),
            (2, crate::site::BedAgeUnit::Legacy20HzSteps)
                | (3..=11, crate::site::BedAgeUnit::Microseconds)
        ) {
            return Err("Invalid saved terrain clock".into());
        }
        let carried_water = s
            .active_soil
            .as_ref()
            .map_or(0., |s| s.material().pore_water_m3)
            + s.machine.bucket.pore_water_m3
            + s.player.hand.pore_water_m3
            + s.sluice.active_material().pore_water_m3
            + s.sluice.tailings.pore_water_m3
            + s.sluice.concentrate.pore_water_m3
            + s.falling
                .iter()
                .map(|l| l.material.pore_water_m3)
                .sum::<f64>();
        if s.version < 4
            && (s.site.soil_water_m3() > 0.
                || carried_water > 0.
                || s.sluice.feed_water_credit_m3 > 0.)
        {
            return Err("Legacy save contains unsupported soil water".into());
        }
        if s.version < 5 && (s.site.bed_remainder_m3() > 0. || s.site.has_projection_inventory()) {
            return Err("Legacy save contains unsupported subgrid bed inventory".into());
        }
        let advanced = s
            .machine
            .articulation
            .as_ref()
            .is_some_and(|a| a.payload_moments.is_some() || a.exchange.is_some())
            || s.falling.iter().any(|l| l.rigid.is_some())
            || s.material_motion.is_some();
        let shell = s
            .machine
            .articulation
            .as_ref()
            .is_some_and(|a| a.bucket_model().is_shell());
        if (s.version == 10) != (s.machine.bucket_model() == crate::bucket::Model::ShellV1)
            || (s.version == 11)
                != (s.machine.bucket_model() == crate::bucket::Model::MountedShellV2)
            || (s.version >= 10 && !shell)
            || (s.version < 7 && s.machine.articulation.is_some())
            || (s.version == 7 && s.machine.articulation.is_none())
            || (s.version < 8 && advanced)
            || (s.version == 8 && !advanced)
            || (s.version < 10 && (s.version == 9) != s.active_soil.is_some())
            || (s.active_soil.is_some() && s.machine.articulation.is_none())
        {
            return Err("Saved material dynamics do not match the format version".into());
        }
        if s.material_motion.as_ref().is_some_and(|m| !m.valid()) {
            return Err("Invalid material boundary receipt".into());
        }
        if s.version < 6 && s.machine.tracks.carries_motion() {
            return Err("Legacy save contains unsupported machine dynamics".into());
        }
        if s.active_soil.as_ref().is_some_and(|soil| !soil.valid()) {
            return Err("Invalid saved moving soil".into());
        }
        let site = Site::restore(s.site)?;
        if s.active_soil
            .as_ref()
            .is_some_and(|soil| !soil.dry_contact(&site))
        {
            return Err("Saved moving soil has unsupported wet contact".into());
        }
        if !s.machine.valid()
            || !s.sluice.valid()
            || !s.player.position.is_finite()
            || !Site::bounds(s.player.position, 0.0)
            || !s.player.yaw.is_finite()
            || !s.player.pitch.is_finite()
            || s.player.pitch.abs() > 1.5
            || !s.player.hand.valid()
            || !s.player.hand.volume().is_finite()
            || !s.player.tool_swing.is_finite()
            || !(0.0..=1.0).contains(&s.player.tool_swing)
            || !s.sim_time.is_finite()
            || s.sim_time < 0.0
            || !s.initial_gold_g.is_finite()
            || s.initial_gold_g < 0.0
            || !s.delivered_m3.is_finite()
            || s.delivered_m3 < 0.0
            || s.falling.len() > 2048
            || s.falling.iter().any(|l| {
                !l.position.is_finite()
                    || !l.velocity.is_finite()
                    || !l.material.valid()
                    || !l.valid_rigid_state()
                    || l.position.length() > 1000.0
                    || l.velocity.length() > 100.0
            })
        {
            return Err("Invalid player, machine or material state".into());
        }
        if let Some(p) = s.sluice.placement {
            // Placement may be unsupported after later excavation, but the saved
            // transform and angle must still be finite and in the claim.
            if !Site::bounds(p.inlet, 0.0)
                || !p.yaw.is_finite()
                || !(3.0..=12.0).contains(&p.slope_deg)
            {
                return Err("Invalid saved placement".into());
            }
        }
        let water_balance = s.water_balance.unwrap_or_else(|| {
            crate::water_balance::WaterBalance::new(
                site.water_volume_m3()
                    + site.soil_water.stored_m3()
                    + s.sluice.stored_water_m3()
                    + carried_water,
                s.sim_time,
                true,
            )
        });
        if !water_balance.valid() {
            return Err("Invalid water ledger".into());
        }
        let session = Self {
            site,
            machine: s.machine,
            sluice: s.sluice,
            player: s.player,
            falling: s.falling,
            sim_time: s.sim_time,
            paused: false,
            message: "Shift restored".into(),
            initial_gold_g: s.initial_gold_g,
            delivered_m3: s.delivered_m3,
            water_balance,
            material_motion: s.material_motion,
            active_soil: s.active_soil,
        };
        if (session.accounted_gold() - session.initial_gold_g).abs() > 1e-3 {
            return Err("Save has an inconsistent gold ledger".into());
        }
        Ok(session)
    }
    pub fn stored_water_m3(&self) -> f64 {
        self.site.water_volume_m3()
            + self
                .active_soil
                .as_ref()
                .map_or(0., |s| s.material().pore_water_m3)
            + self.site.soil_water.stored_m3()
            + self.sluice.stored_water_m3()
            + self.machine.bucket.pore_water_m3
            + self.player.hand.pore_water_m3
            + self.sluice.active_material().pore_water_m3
            + self.sluice.tailings.pore_water_m3
            + self.sluice.concentrate.pore_water_m3
            + self
                .falling
                .iter()
                .map(|l| l.material.pore_water_m3)
                .sum::<f64>()
    }
    pub fn water_balance_error_m3(&self) -> f64 {
        self.water_balance.error_m3(self.stored_water_m3())
    }
}
