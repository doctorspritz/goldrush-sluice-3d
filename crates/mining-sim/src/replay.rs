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
    dig_position_z: f32,
    dump_position_z: f32,
    lift_stage: u8,
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
                self.dump_position_z = s.machine.position.z;
                self.command(s, Action::EnterExit);
                if s.player.third_person {
                    self.command(s, Action::Camera);
                }
                self.next();
            }
            2 | 10 => {
                if self.ticks == 1 {
                    self.dig_position_z = s.machine.position.z;
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
                if self.ticks == 1 {
                    self.lift_stage = 0;
                }
                // Lift the loaded, closed bucket clear before opening/retracting
                // it. Simultaneous old joint targets drove the heel back into
                // the bank once actual chassis/contact reactions were present.
                if self.lift_stage == 0 {
                    controls.machine.boom = axis(1.1 - s.machine.boom, 0.42);
                    if controls.machine.boom == 0. {
                        self.lift_stage = 1;
                    }
                } else if self.lift_stage == 1 {
                    if seek_arm(s, &mut controls.machine, 1.0, -1.65, -0.2) {
                        self.lift_stage = 2;
                    }
                } else if s.machine.position.z > self.dump_position_z + 0.05 {
                    // Carry the closed load back to the receiving pad instead
                    // of extending beyond reach as the excavation face advances.
                    controls.machine.drive = -1.;
                    controls.machine.steer = (-s.machine.heading).sin() * 2.;
                } else {
                    self.next();
                }
            }
            4 | 12 => {
                if seek_hopper(s, &mut controls.machine) {
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
                    if self.phase == 13 && s.delivered_m3 < 0.4 {
                        // Actual terrain evolution changes the volume of a
                        // pass. Target delivered material, with a finite guard,
                        // instead of assuming four scoops always suffice.
                        if self.loads >= 12 {
                            self.error = Some(format!(
                                "Twelve finite loads delivered only {:.3} m3",
                                s.delivered_m3
                            ));
                        } else {
                            self.phase = 6;
                            self.ticks = 0;
                        }
                    } else {
                        self.next();
                    }
                }
            }
            6 => {
                if self.ticks == 1 {
                    self.drive_target_z = self.dig_position_z + 0.54;
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
                controls.machine.steer = (-s.machine.heading).sin() * 2.;
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

// Aim the real dump mouth above the finite hopper with ordinary joint axes.
// Targets depend on the tilted chassis and current reach, rather than assuming
// the same fixed transport angles reach the hopper after every advance.
fn seek_hopper(s: &Session, i: &mut MachineInput) -> bool {
    let target = Vec3::new(
        crate::sluice::HOPPER_X,
        crate::Sluice::hopper_top(&s.site) + 1.05,
        crate::sluice::HOPPER_Z,
    );
    let local = s.machine.chassis_rotation().inverse() * (target - s.machine.position);
    if !seek_swing(s, i, local.x.atan2(local.z)) {
        return false;
    }
    // shoulder + dump offsets are 0.70 m forward and 1.55 m up.
    let radial = local.x.hypot(local.z) - 0.70;
    let height = local.y - 1.55;
    let a = crate::machine::BOOM_LENGTH;
    let b = crate::machine::STICK_LENGTH;
    let stick = -((radial * radial + height * height - a * a - b * b) / (2. * a * b))
        .clamp(-1., 1.)
        .acos();
    let boom = height.atan2(radial) - (b * stick.sin()).atan2(a + b * stick.cos());
    seek_arm(
        s,
        i,
        boom.clamp(-0.24, 1.20),
        stick.clamp(-2.15, -0.15),
        -0.2,
    )
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

/// Bounded validation stroke for the prepared drained-soil bank. This supplies
/// ordinary joystick axes only: no pose, material or contact edits. A grade
/// controller keeps actual steel clear of the lowered, still-solid substrate.
pub fn soil_trial_controls(s: &Session, elapsed: f64) -> MachineInput {
    if s.active_soil.is_none() || !s.machine.bucket_model().is_shell() {
        return MachineInput::default();
    }
    let arm = s.machine.arm();
    let q = s.machine.bucket_rotation();
    let mounted = s.machine.bucket_model() == crate::bucket::Model::MountedShellV2;
    let clearance = s
        .machine
        .bucket_model()
        .geometry()
        .parts
        .iter()
        .flat_map(|p| &p.vertices)
        .map(|v| arm.wrist + q * *v)
        .map(|p| {
            p.y - if mounted {
                crate::soil::substrate_height(&s.site, p.x, p.z)
            } else {
                s.site.height(p.x, p.z)
            }
        })
        .fold(f32::INFINITY, f32::min);
    let grade: f32 = if clearance > 0.04 {
        -1.
    } else if clearance < 0.015 {
        1.
    } else {
        0.
    };
    if mounted {
        let (rake_angle, mouth_pitch) = soil_trial_world_angles(&s.machine);
        let entry_rake = soil_trial_entry_rake(s);
        let aligned = (rake_angle - entry_rake).abs() <= 0.05;
        // The steel meets the top of the clods before reaching the cleared
        // substrate. Begin drawing on measured contact instead of continuing
        // to press downward until the chassis jacks up. Actual stick progress
        // retains that phase if the reaction unloads during entry. The 10% force
        // cue and 0.50 rad draw are demo operator choices, not material laws.
        let entered = clearance <= 0.04
            || s.machine.stick < -1.31
            || s.machine.tracks.feedback.bucket_reaction_n.length()
                > s.machine.hydraulics.bucket_force_n * 0.1;
        let working_grade = if entered { grade.max(0.) } else { grade };
        return if elapsed < 0.3 {
            MachineInput::default()
        } else if elapsed < 6.5 && s.machine.stick > -1.80 {
            // Align the actual teeth to the bank, draw through a working
            // stroke, then curl. A short 0.15 rad draw only skimmed this bank;
            // the replay's 0.50 rad stroke uses ordinary rate-limited axes.
            MachineInput {
                stick: if entered && aligned { -0.7 } else { 0. },
                boom: if aligned {
                    working_grade
                } else {
                    grade.max(0.)
                },
                curl: if rake_angle < entry_rake - 0.05 {
                    -1.
                } else if rake_angle > entry_rake + 0.05 {
                    1.
                } else {
                    0.
                },
                ..Default::default()
            }
        } else if elapsed < 6.5 {
            MachineInput {
                curl: 1.,
                boom: working_grade,
                ..Default::default()
            }
        } else if elapsed < 9.0 {
            MachineInput {
                boom: 1.,
                curl: if mouth_pitch > 0.1 { 1. } else { 0. },
                ..Default::default()
            }
        } else if elapsed < 10.0 {
            MachineInput::default()
        } else if elapsed < 16.5 {
            MachineInput {
                curl: -1.,
                ..Default::default()
            }
        } else {
            MachineInput::default()
        };
    }
    if elapsed < 0.3 {
        MachineInput::default()
    } else if elapsed < 2.0 {
        MachineInput {
            stick: -0.7,
            curl: 0.25,
            boom: grade,
            ..Default::default()
        }
    } else if elapsed < 3.7 {
        MachineInput {
            curl: 1.,
            boom: grade,
            ..Default::default()
        }
    } else if elapsed < 5.7 {
        MachineInput {
            boom: 1.,
            ..Default::default()
        }
    } else if elapsed < 6.3 {
        MachineInput::default()
    } else if elapsed < 10.5 {
        MachineInput {
            curl: -1.,
            ..Default::default()
        }
    } else {
        MachineInput::default()
    }
}

// Match the actual cutting tangent to the bank in the horizontal work plane.
// A fixed chassis-frame bucket angle left the teeth above the belly on a slope.
// The 0.08 rad attack is an operator's chosen shallow cut, not a soil parameter.
fn soil_trial_entry_rake(s: &Session) -> f32 {
    let arm = s.machine.arm();
    let f = Vec3::new(arm.forward.x, 0., arm.forward.z).normalize_or_zero();
    let step = crate::site::CELL * 0.5;
    let a = arm.lip - f * step;
    let b = arm.lip + f * step;
    let grade = (crate::soil::substrate_height(&s.site, b.x, b.z)
        - crate::soil::substrate_height(&s.site, a.x, a.z))
    .atan2(2. * step);
    let tooth = s.machine.bucket_model().geometry().cutting_direction();
    tooth.y.atan2(tooth.z) + grade + 0.08
}

// The bucket rake and mouth tilt are measured against gravity in the arm's
// horizontal work plane. Joint angles alone omit chassis pitch; chasing that
// relative angle opened the teeth toward vertical as a loaded chassis tipped.
fn soil_trial_world_angles(machine: &crate::Excavator) -> (f32, f32) {
    let arm = machine.arm();
    let forward = Vec3::new(arm.forward.x, 0., arm.forward.z).normalize_or_zero();
    let rotation = machine.bucket_rotation();
    let rake = rotation * -Vec3::Z;
    let mouth = rotation * machine.bucket_model().geometry().mouth_normal();
    (
        rake.y.atan2(rake.dot(forward)),
        (-mouth.dot(forward)).atan2(mouth.y),
    )
}

#[cfg(test)]
mod soil_controller_tests {
    use super::*;
    #[test]
    fn bank_entry_puts_the_teeth_at_the_cutting_surface_on_the_actual_slope() {
        let mut s = Session::new();
        s.prepare_soil_trial().unwrap();
        let target = soil_trial_entry_rake(&s);
        let gap = |s: &Session| {
            let arm = s.machine.arm();
            let q = s.machine.bucket_rotation();
            let mut tooth = f32::INFINITY;
            let mut shell = f32::INFINITY;
            for part in &s.machine.bucket_model().geometry().parts {
                for v in &part.vertices {
                    let p = arm.wrist + q * *v;
                    let clearance = p.y - crate::soil::substrate_height(&s.site, p.x, p.z);
                    if part.tooth {
                        tooth = tooth.min(clearance)
                    } else {
                        shell = shell.min(clearance)
                    }
                }
            }
            tooth - shell
        };
        s.machine.curl = s.machine.boom + s.machine.stick + std::f32::consts::FRAC_PI_2 - 1.1;
        let old_gap = gap(&s);
        assert!(
            old_gap > 0.02,
            "sloping-bank defect was not reproduced: {old_gap}"
        );
        s.machine.curl = s.machine.boom + s.machine.stick + std::f32::consts::FRAC_PI_2 - target;
        let corrected_gap = gap(&s);
        assert!(
            corrected_gap < 0.01,
            "the belly still reaches soil before the teeth: {corrected_gap}"
        );
        println!("old tooth above first shell contact={old_gap}m, bank-aligned={corrected_gap}m, entry rake={target}rad");
        let g = s.machine.bucket_model().geometry();
        let direction = g.cutting_direction();
        // Independent tangent of the declared circular floor's final segment.
        let theta = 1.1 - 1.1 / 16.;
        assert!(direction.distance(Vec3::new(0., f32::sin(theta), f32::cos(theta))) < 1e-6);
    }
    #[test]
    fn bank_contact_starts_drawing_before_closing_and_releases_downward_input() {
        let mut s = Session::new();
        s.prepare_soil_trial().unwrap();
        let entry_rake = soil_trial_entry_rake(&s);
        s.machine.articulation = None;
        s.machine.curl =
            s.machine.boom + s.machine.stick + std::f32::consts::FRAC_PI_2 - entry_rake;
        s.machine.enable_articulation().unwrap();
        let approach = soil_trial_controls(&s, 0.4);
        assert!(approach.boom < 0. && approach.stick == 0. && approach.curl == 0.);
        s.machine.tracks.feedback.bucket_reaction_n =
            Vec3::Y * (s.machine.hydraulics.bucket_force_n * 0.11);
        let drawing = soil_trial_controls(&s, 0.5);
        assert!(
            drawing.stick < 0. && drawing.boom >= 0. && drawing.curl == 0.,
            "contact must begin the draw, not close the bucket before the teeth enter"
        );
        s.machine.articulation = None;
        s.machine.stick = -1.32;
        s.machine.curl =
            s.machine.boom + s.machine.stick + std::f32::consts::FRAC_PI_2 - entry_rake;
        s.machine.enable_articulation().unwrap();
        s.machine.tracks.feedback.bucket_reaction_n = Vec3::ZERO;
        let released = soil_trial_controls(&s, 0.6);
        assert!(
            released.stick < 0. && released.boom >= 0. && released.curl == 0.,
            "contact unloading restarted the downward entry phase"
        );
    }
    #[test]
    fn bank_entry_uses_world_rake_when_the_chassis_pitches() {
        for yaw in [0., 0.7, 2.4] {
            for pitch in [-0.3, 0., 0.6] {
                let mut s = Session::new();
                s.prepare_soil_trial().unwrap();
                s.machine.articulation = None;
                s.machine.heading = yaw;
                s.machine.position.y += 10.;
                s.machine.tracks.tilt = glam::Quat::from_rotation_x(-pitch);
                s.machine.enable_articulation().unwrap();
                assert!(s.machine.articulation.as_ref().unwrap().valid());
                let relative = s.machine.arm().angle;
                let (rake, mouth) = soil_trial_world_angles(&s.machine);
                let n = s.machine.bucket_model().geometry().mouth_normal();
                assert!(
                    (rake - (relative + pitch)).abs() < 5e-6,
                    "yaw={yaw}, pitch={pitch}, rake={rake}, relative={relative}"
                );
                assert!((mouth - (rake + n.z.atan2(n.y))).abs() < 5e-6);
                if pitch > 0.5 {
                    assert!(
                        relative < 1.05,
                        "fixture no longer reproduces the old opening command"
                    );
                    assert!(
                        soil_trial_controls(&s, 0.4).curl > 0.,
                        "tilted teeth kept opening toward vertical"
                    );
                }
            }
        }
    }
}
