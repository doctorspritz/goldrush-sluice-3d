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
        let water_balance =
            crate::water_balance::WaterBalance::new(site.water_volume_m3(), 0., false);
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
                        "Cab controls: R/F boom, T/G stick, Y/H curl. Hold Space to cut; J pours the bucket.".into()
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
        if self.player.mode == ControlMode::Excavator {
            // Equipment, rather than camera focus, owns the running simulation.
            let old_yaw = self.machine.heading + self.machine.swing;
            let old_position = self.machine.position;
            if let Some((position, velocity, material)) =
                self.machine.step(input.machine, &mut self.site, dt)
            {
                self.falling.push(FallingLoad {
                    position,
                    velocity,
                    material,
                });
            }
            let p = self.machine.position;
            let hopper = Sluice::hopper_position(&self.site);
            let blocks_hopper = Vec3::new(p.x - hopper.x, 0.0, p.z - hopper.z).length() < 2.2;
            let blocks_sluice = self.sluice.placement.is_some_and(|placement| {
                let d = p - placement.inlet;
                let x = d.dot(placement.forward());
                let z = d.dot(placement.right()).abs();
                (-1.8..self.sluice.profile.length() + 1.8).contains(&x) && z < 1.7
            });
            if blocks_hopper || blocks_sluice {
                self.machine.position = old_position;
            }
            self.player.yaw += self.machine.heading + self.machine.swing - old_yaw;
            self.player.position = self.machine.position + Vec3::Y * 1.9;
        } else {
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
        for (i, load) in self.falling.iter_mut().enumerate() {
            let before = load.position;
            load.velocity.y -= 9.81 * dt;
            load.position += load.velocity * dt;
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
        let requested = if self.sluice.pump_on && self.sluice.placement.is_some() {
            let room = self
                .sluice
                .profile
                .capacity()
                .saturating_sub(self.sluice.water.len() + self.sluice.outflow.len())
                as f64
                * self.sluice.profile.particle_m3()
                - self.sluice.water_credit_m3;
            (self.sluice.flow_m3s * dt).min(room.max(0.0) as f32)
        } else {
            0.0
        };
        let actual = if requested > 0.0 {
            self.site.withdraw_water(
                Vec3::new(Site::river_center(HOPPER_Z), 0.0, HOPPER_Z),
                requested,
            )
        } else {
            0.0
        };
        let expected = self.sluice.flow_m3s * dt;
        let fraction = if expected > 0.0 {
            actual / expected
        } else {
            0.0
        };
        self.sluice.water_used_m3 += actual as f64;
        self.sluice.feed(dt, fraction > 0.01);
        self.sluice.water_credit_m3 += actual as f64;
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
    }
    pub fn save_json(&self) -> Result<String, String> {
        serde_json::to_string(&Save {
            version: 2,
            site: self.site.snapshot(),
            machine: self.machine.clone(),
            sluice: self.sluice.clone(),
            player: self.player.clone(),
            falling: self.falling.clone(),
            sim_time: self.sim_time,
            initial_gold_g: self.initial_gold_g,
            delivered_m3: self.delivered_m3,
            water_balance: Some(self.water_balance.clone()),
        })
        .map_err(|e| e.to_string())
    }
    pub fn load_json(json: &str) -> Result<Self, String> {
        if json.len() > 128_000_000 {
            return Err("Save file is too large".into());
        }
        let s: Save = serde_json::from_str(json).map_err(|e| e.to_string())?;
        if s.version != 2 {
            return Err("Unsupported save version".into());
        }
        let site = Site::restore(s.site)?;
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
                site.water_volume_m3() + s.sluice.stored_water_m3(),
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
        };
        if (session.accounted_gold() - session.initial_gold_g).abs() > 1e-3 {
            return Err("Save has an inconsistent gold ledger".into());
        }
        Ok(session)
    }
    pub fn stored_water_m3(&self) -> f64 {
        self.site.water_volume_m3() + self.sluice.stored_water_m3()
    }
    pub fn water_balance_error_m3(&self) -> f64 {
        self.water_balance.error_m3(self.stored_water_m3())
    }
}
