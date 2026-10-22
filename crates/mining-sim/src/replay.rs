//! Deterministic operator replay for the full loop. It issues the same commands and
//! joint axes as a player; it never injects material or teleports equipment.
use crate::{session::Controls, Action, ControlMode, MachineInput, Placement, Session, Vec3};

#[derive(Clone, Default)]
pub struct OperatorReplay {
    pub phase: u32,
    pub ticks: u32,
    pub complete: bool,
    pub error: Option<String>,
    inspection_step: u8,
    dig_step: u8,
    dig_started: u32,
    dig_height: f32,
    drive_target_z: f32,
    loads: u32,
}
impl OperatorReplay {
    fn next(&mut self) {
        self.phase += 1;
        self.ticks = 0;
    }
    fn command(&mut self, s: &mut Session, a: Action) {
        if let Err(e) = s.on_action(a) {
            self.error = Some(format!("Phase {}: {e}", self.phase));
        }
    }
    pub fn controls(&mut self, s: &mut Session) -> Controls {
        if self.complete || self.error.is_some() {
            return Controls::default();
        }
        self.ticks += 1;
        let mut controls = Controls::default();
        match self.phase {
            0 => {
                if self.ticks > 90 {
                    self.next();
                }
            }
            1 => {
                self.command(s, Action::EnterExit);
                if s.player.third_person {
                    self.command(s, Action::Camera);
                }
                self.next();
            }
            2 | 10 => {
                if self.ticks == 1 {
                    self.dig_step = 0;
                    self.dig_started = 1;
                    let lip = s.machine.arm().lip;
                    self.dig_height =
                        (s.site.height(lip.x, lip.z) - 0.4).max(s.site.bedrock_height(lip) + 0.03);
                }
                controls.machine.cut = true;
                if self.dig_step == 0 {
                    controls.machine.boom = -0.3;
                    if s.machine.arm().lip.y <= self.dig_height + 0.01 || self.ticks >= 120 {
                        self.dig_step = 1;
                        self.dig_started = self.ticks;
                    }
                } else {
                    controls.machine.stick = if s.machine.stick > -2.1 { -0.45 } else { 0. };
                    controls.machine.curl = if s.machine.curl < 0.9 { 0.4 } else { 0. };
                    // Coordinate ordinary joint axes to crowd/curl along the
                    // cut. This is a replay controller, not prescribed soil yield.
                    let lip = s.machine.arm().lip;
                    let mut shifted = s.machine.clone();
                    shifted.boom += 0.001;
                    let boom_gain = (shifted.arm().lip.y - lip.y) / 0.001 * 0.42;
                    shifted = s.machine.clone();
                    shifted.stick += 0.001;
                    let stick_gain = (shifted.arm().lip.y - lip.y) / 0.001 * 0.55;
                    shifted = s.machine.clone();
                    shifted.curl += 0.001;
                    let curl_gain = (shifted.arm().lip.y - lip.y) / 0.001 * 0.9;
                    let desired = (self.dig_height - lip.y) * 6.
                        - stick_gain * controls.machine.stick
                        - curl_gain * controls.machine.curl;
                    controls.machine.boom = (desired / boom_gain.max(0.1)).clamp(-1., 1.);
                }
                if s.machine.bucket.volume() >= 0.265
                    || (self.dig_step == 1 && self.ticks - self.dig_started >= 240)
                {
                    self.next();
                }
            }
            3 | 11 => {
                if seek_arm(s, &mut controls.machine, 1.0, -1.65, -0.2) {
                    self.next();
                }
            }
            4 | 12 => {
                let target = Vec3::new(crate::sluice::HOPPER_X, 0., crate::sluice::HOPPER_Z)
                    - s.machine.position;
                if seek_swing(
                    s,
                    &mut controls.machine,
                    target.x.atan2(target.z) - s.machine.heading,
                ) {
                    self.next();
                }
            }
            5 | 13 => {
                if self.ticks == 1 && !s.player.third_person {
                    self.command(s, Action::Camera);
                }
                controls.machine.dump = true;
                if self.ticks >= 100 && s.machine.bucket.is_empty() && s.falling.is_empty() {
                    self.loads += 1;
                    if self.phase == 13 && s.delivered_m3 < 0.4 && self.loads < 4 {
                        self.phase = 6;
                        self.ticks = 0;
                    } else {
                        self.next();
                    }
                }
            }
            6 => {
                if self.ticks == 1 {
                    self.drive_target_z = s.machine.position.z + 0.54;
                }
                if seek_arm(s, &mut controls.machine, 0.95, -1.3, -0.2) {
                    self.next();
                }
            }
            7 => {
                if seek_swing(s, &mut controls.machine, 0.0) {
                    self.next();
                }
            }
            8 => {
                controls.machine.drive = 1.0;
                if s.machine.position.z >= self.drive_target_z {
                    controls.machine.drive = 0.;
                    self.next();
                }
            }
            9 => {
                if seek_arm(s, &mut controls.machine, 0.48, -1.30, -0.20)
                    || matches!(s.machine.digging.reason, crate::machine::DigLimit::Ground)
                {
                    self.next();
                }
            }
            14 => {
                self.command(s, Action::EnterExit);
                self.next();
            }
            15..=17 => {
                let target = match self.phase {
                    15 => Vec3::new(21.6, 0.0, 14.0),
                    16 => Vec3::new(14.6, 0.0, 14.0),
                    _ => Vec3::new(14.6, 0.0, 15.0),
                };
                let d = target - Vec3::new(s.player.position.x, 0.0, s.player.position.z);
                s.player.yaw = d.x.atan2(d.z);
                s.player.pitch = -0.24;
                controls.walk = 1.0;
                controls.sprint = true;
                if d.length() < 0.12 {
                    controls.walk = 0.0;
                    self.next();
                }
            }
            18 => {
                self.command(s, Action::Place(Placement::suggested(&s.site)));
                self.next();
            }
            19 => {
                self.command(s, Action::TogglePump);
                self.command(s, Action::ToggleFeed);
                self.next();
            }
            20 => {
                if self.inspection_step < 2 {
                    let target = if self.inspection_step == 0 {
                        Vec3::new(16.6, 0.0, 15.0)
                    } else {
                        Vec3::new(16.6, 0.0, 20.5)
                    };
                    let d = target - Vec3::new(s.player.position.x, 0.0, s.player.position.z);
                    s.player.yaw = d.x.atan2(d.z);
                    controls.walk = 1.0;
                    if d.length() < 0.1 {
                        self.inspection_step += 1;
                        controls.walk = 0.0;
                    }
                } else {
                    let d = Vec3::new(14.6, 0.0, 19.6)
                        - Vec3::new(s.player.position.x, 0.0, s.player.position.z);
                    s.player.yaw = d.x.atan2(d.z);
                    s.player.pitch = -0.58;
                    if s.player.third_person {
                        self.command(s, Action::Camera);
                    }
                }
                if s.sluice.hopper.is_empty() && s.falling.is_empty() {
                    self.command(s, Action::ToggleFeed);
                    self.next();
                }
            }
            21 => {
                if self.ticks > 900 {
                    self.command(s, Action::TogglePump);
                    self.command(s, Action::Cleanup);
                    self.next();
                }
            }
            _ => {
                self.complete = true;
            }
        }
        if self.ticks > 15_000 {
            self.error = Some(format!("Operator replay stuck in phase {}", self.phase));
        }
        controls
    }
    pub fn label(&self) -> &'static str {
        match self.phase {
            0 => "site",
            1..=3 => "dig",
            4..=5 => "deliver",
            6..=10 => "next dig",
            11..=13 => "next delivery",
            14..=17 => "walk to plant",
            18 => "place sluice",
            19..=20 => "wash",
            21 => "clear and collect",
            _ => "complete",
        }
    }
}
fn axis(error: f32, rate: f32) -> f32 {
    if error.abs() < 0.008 {
        0.0
    } else {
        (error / rate * 60.0).clamp(-1.0, 1.0)
    }
}
fn seek_arm(s: &Session, i: &mut MachineInput, boom: f32, stick: f32, curl: f32) -> bool {
    i.boom = axis(boom - s.machine.boom, 0.42);
    i.stick = axis(stick - s.machine.stick, 0.55);
    i.curl = axis(curl - s.machine.curl, 0.90);
    i.boom == 0.0 && i.stick == 0.0 && i.curl == 0.0
}
fn seek_swing(s: &Session, i: &mut MachineInput, target: f32) -> bool {
    let d = target - s.machine.swing;
    let error = d.sin().atan2(d.cos());
    i.swing = axis(error, 0.65);
    i.swing == 0.0
}

pub fn replay_receipt(s: &Session, r: &OperatorReplay) -> Result<String, String> {
    if let Some(e) = &r.error {
        return Err(e.clone());
    }
    if !r.complete {
        return Err("Replay did not complete".into());
    }
    if s.site.excavated.volume() < 0.4 {
        return Err("Insufficient actual excavation".into());
    }
    if s.delivered_m3 < 0.35 {
        return Err(format!(
            "Loads missed the hopper: delivered {:.3} m3",
            s.delivered_m3
        ));
    }
    if s.sluice.concentrate.gold_g < crate::session::GOLD_TARGET_G {
        return Err("The first-gold objective was not reached".into());
    }
    if s.sluice.water_returned_m3 < 0.01 {
        return Err("Water did not flow through the sluice back to terrain".into());
    }
    if s.sluice.tailings.volume() < 0.1 {
        return Err("Gangue did not leave the plant".into());
    }
    if (s.accounted_gold() - s.initial_gold_g).abs() > 1e-3 {
        return Err("Gold ledger did not balance".into());
    }
    if s.player.mode != ControlMode::OnFoot {
        return Err("Operator did not leave the cab".into());
    }
    serde_json::to_string_pretty(&serde_json::json!({"complete":true,"simulated_seconds":s.sim_time,
        "excavated_m3":s.site.excavated.volume(),"delivered_m3":s.delivered_m3,"processed_m3":s.sluice.processed_m3,
        "gold_recovered_g":s.sluice.concentrate.gold_g,"gold_lost_g":s.sluice.tailings.gold_g,
        "tailings_m3":s.sluice.tailings.volume(),"ledger_error_g":s.accounted_gold()-s.initial_gold_g,
        "water_used_m3":s.sluice.water_used_m3})).map_err(|e|e.to_string())
}

/// Fresh-claim excavation/delivery gate. This deliberately reports its phase
/// scope: reaching the hopper does not establish washing or gold recovery.
pub fn earthworks_receipt(s: &Session, r: &OperatorReplay) -> Result<String, String> {
    if let Some(error) = &r.error {
        return Err(error.clone());
    }
    if r.phase < 15 || s.player.mode != ControlMode::OnFoot {
        return Err("Earthwork replay did not leave the cab after delivery".into());
    }
    if s.delivered_m3 < 0.4 || !s.machine.bucket.is_empty() || !s.falling.is_empty() {
        return Err("Finite excavation/delivery was incomplete".into());
    }
    let fresh = Session::new();
    let initial = fresh.accounted_bulk_material_m3();
    let bedrock_source = (fresh
        .site
        .world
        .bedrock_elevation
        .iter()
        .map(|v| *v as f64)
        .sum::<f64>()
        - s.site
            .world
            .bedrock_elevation
            .iter()
            .map(|v| *v as f64)
            .sum::<f64>())
        * (crate::site::CELL as f64).powi(2);
    let solid_error = s.accounted_bulk_material_m3() - initial - bedrock_source;
    let gold_error = s.accounted_gold() - s.initial_gold_g;
    let water_error = s.water_balance_error_m3();
    // One ppm of the full claim's bulk inventory is the reporting tolerance
    // for GPU terrain transfers. Local cut tests use a separate 1e-6 m3 gate.
    let solid_tolerance = initial * 1e-6;
    if !solid_error.is_finite() || solid_error.abs() > solid_tolerance {
        return Err(format!(
            "Whole-claim solid residual {solid_error} m3 exceeds {solid_tolerance}"
        ));
    }
    if !gold_error.is_finite()
        || gold_error.abs() > 1e-6
        || !water_error.is_finite()
        || water_error.abs() > 0.00005
    {
        return Err(format!(
            "Earthwork ledger residuals: gold {gold_error} g, water {water_error} m3"
        ));
    }
    serde_json::to_string_pretty(&serde_json::json!({"verified_phase":"fresh-claim excavation and delivery",
        "full_mining_loop_verified":false,"simulated_seconds":s.sim_time,"excavated_m3":s.site.excavated.volume(),
        "delivered_m3":s.delivered_m3,"hopper_m3":s.sluice.hopper.volume(),"hopper_gold_g":s.sluice.hopper.gold_g,
        "initial_bulk_inventory_m3":initial,"bedrock_change_as_bulk_source_m3":bedrock_source,"solid_balance_error_m3":solid_error,"solid_tolerance_m3":solid_tolerance,
        "water_balance_error_m3":water_error,"gold_balance_error_g":gold_error,"machine":s.machine,
        "soil_model":"25 cm heightfield, swept floor geometry, generic passive blade pressure; not calibrated continuum soil or machine dynamics"})).map_err(|e|e.to_string())
}
