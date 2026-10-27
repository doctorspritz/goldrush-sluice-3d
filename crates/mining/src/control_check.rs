//! Native regression through Bevy's keyboard/mouse and toolbar handlers. Inputs
//! are synthetic ECS events, not OS key injection or a manual playtest.
use crate::{app::Game, hud::HudButton, validation::Validation};
use bevy::{
    input::mouse::AccumulatedMouseMotion,
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use mining_sim::ControlMode;
use std::path::PathBuf;

#[derive(Resource, Default)]
pub struct Check {
    path: Option<PathBuf>,
    frame: u64,
    checks: Vec<&'static str>,
    yaw: f32,
    position: Vec3,
    joints: [f32; 3],
    clock: f64,
    finished: bool,
    mechanism_motion: bool,
    payload_motion: bool,
    soil_motion: bool,
    soil_saved: bool,
    screen_mode: bool,
    screen_ready: bool,
    screen_before: Option<(mining_sim::Material, mining_sim::Material)>,
    screen_saved: Option<serde_json::Value>,
    screen_stages: Vec<serde_json::Value>,
}
impl Check {
    pub fn from_args() -> Self {
        let args: Vec<_> = std::env::args().collect();
        let mechanism_motion = args.iter().any(|a| a == "--mechanism-motion");
        assert!(
            !mechanism_motion
                || (args.iter().any(|a| a == "--articulated-machine")
                    && args.iter().any(|a| a == "--capture-motion")),
            "mechanism-motion requires the articulated model and bounded motion capture"
        );
        assert!(
            !args.iter().any(|a| a == "--soil-motion")
                || ["--active-soil", "--new-shift", "--capture-motion"]
                    .iter()
                    .all(|flag| args.iter().any(|a| a == flag)),
            "soil-motion requires finite active-soil setup and bounded capture"
        );
        Self {
            mechanism_motion,
            payload_motion: args.iter().any(|a| a == "--payload-motion"),
            soil_motion: args.iter().any(|a| a == "--soil-motion"),
            screen_mode: args.iter().any(|a| a == "--check-screen"),
            path: args
                .iter()
                .position(|a| matches!(a.as_str(), "--check-controls" | "--check-screen"))
                .and_then(|i| args.get(i + 1))
                .map(PathBuf::from),
            ..default()
        }
    }
    fn expect(&mut self, condition: bool, label: &'static str, validation: &mut Validation) {
        if condition {
            self.checks.push(label);
        } else {
            validation.failure = Some(format!("Native input check failed: {label}"));
        }
    }
}
pub fn inject(
    mut check: ResMut<Check>,
    mut game: ResMut<Game>,
    mut keys: ResMut<ButtonInput<KeyCode>>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    mut motion: ResMut<AccumulatedMouseMotion>,
    mut buttons: Query<(&HudButton, &mut Interaction)>,
    validation: Res<Validation>,
) {
    if (check.path.is_none()
        && !check.mechanism_motion
        && !check.payload_motion
        && !check.soil_motion)
        || check.finished
    {
        return;
    }
    if check.screen_mode && !check.screen_ready {
        if !validation.capture_scene_ready() || !validation.ready() {
            return;
        }
        check.screen_ready = true;
    }
    check.frame += 1;
    if check.screen_mode {
        let key = match check.frame {
            181 | 182 => Some(KeyCode::Escape),
            220 | 221 | 460 | 461 => Some(KeyCode::KeyK),
            260 | 261 => Some(KeyCode::KeyL),
            300 | 301 => Some(KeyCode::KeyM),
            340 | 341 => Some(KeyCode::F5),
            400 | 401 => Some(KeyCode::F9),
            _ => None,
        };
        if let Some(key) = key {
            if matches!(check.frame, 181 | 220 | 260 | 300 | 340 | 400 | 460) {
                keys.press(key);
            } else {
                keys.release(key);
            }
        }
        return;
    }
    if check.soil_motion {
        // Enter through the ordinary key handler; the analog validation stroke
        // below uses the same latched joystick packet and worker as gameplay.
        if check.frame == 2 {
            keys.press(KeyCode::Tab);
        }
        if check.frame == 3 {
            keys.release(KeyCode::Tab);
        }
        keys.release(KeyCode::F5);
        let save_at =
            if game.session.machine.bucket_model() == mining_sim::bucket::Model::MountedShellV2 {
                9.4
            } else {
                5.7
            };
        if game.session.sim_time >= save_at && !check.soil_saved {
            keys.press(KeyCode::F5);
            check.soil_saved = true;
        }
        return;
    }
    if check.payload_motion {
        // Open the bucket until gravity points away from its floor, then
        // exercise separated free-flight events. Contact/retention is not
        // bypassed into ordinary gameplay by this explicit fixture probe.
        let (press, release) = match check.frame {
            181 => (Some(KeyCode::Tab), None),
            182 => (None, Some(KeyCode::Tab)),
            200 => (Some(KeyCode::KeyR), None),
            280 => (Some(KeyCode::KeyH), Some(KeyCode::KeyR)),
            550 => (Some(KeyCode::KeyE), Some(KeyCode::KeyH)),
            670 => (Some(KeyCode::F5), None),
            671 => (None, Some(KeyCode::F5)),
            800 => (None, Some(KeyCode::KeyE)),
            _ => (None, None),
        };
        if let Some(key) = release {
            keys.release(key);
        }
        if let Some(key) = press {
            keys.press(key);
        }
        if check.frame == 650 || check.frame == 730 {
            let m = &game.session.machine;
            let arm = m.arm();
            let normal = arm.up * arm.angle.cos() - arm.forward * arm.angle.sin();
            if normal.y < -0.1 {
                game.command(mining_physics::worker::Command::ReleaseRigidPayload(
                    if check.frame == 650 { 0.5 } else { 1. },
                ));
            } else {
                game.session.message =
                    "Payload probe refused: bucket floor does not clear gravity-driven separation."
                        .into();
            }
        }
        return;
    }
    if check.mechanism_motion {
        // Bounded native evidence uses ordinary Bevy keys. It never resets,
        // pauses, teleports, cuts or injects material into the mechanism.
        let (press, release) = match check.frame {
            181 => (Some(KeyCode::Tab), None),
            182 => (None, Some(KeyCode::Tab)),
            200 => (Some(KeyCode::KeyR), None),
            300 => (Some(KeyCode::KeyT), Some(KeyCode::KeyR)),
            400 => (Some(KeyCode::KeyY), Some(KeyCode::KeyT)),
            500 => (Some(KeyCode::KeyE), Some(KeyCode::KeyY)),
            600 => (Some(KeyCode::KeyF), Some(KeyCode::KeyE)),
            700 => (None, Some(KeyCode::KeyF)),
            _ => (None, None),
        };
        if let Some(key) = release {
            keys.release(key);
        }
        if let Some(key) = press {
            keys.press(key);
        }
        return;
    }
    if check.frame.is_multiple_of(120) {
        println!("NATIVE INPUT frame={}", check.frame);
    }
    motion.delta = Vec2::ZERO;
    // Controlled ECS input keeps the OS cursor free and does not alter window focus.
    match check.frame {
        181 => mouse.press(MouseButton::Left),
        182 => {
            mouse.release(MouseButton::Left);
            motion.delta = Vec2::new(25.0, 0.0);
        }
        190 => keys.press(KeyCode::KeyW),
        191 => keys.release(KeyCode::KeyW),
        221 => keys.press(KeyCode::KeyD),
        222 => keys.release(KeyCode::KeyD),
        240 | 265 => keys.press(KeyCode::F1),
        241 | 266 => keys.release(KeyCode::F1),
        245 => keys.press(KeyCode::KeyW),
        260 => keys.release(KeyCode::KeyW),
        310 | 620 => {
            for (action, mut interaction) in &mut buttons {
                if (check.frame == 310 && matches!(action, HudButton::NewShift))
                    || (check.frame == 620 && matches!(action, HudButton::Demo))
                {
                    *interaction = Interaction::Pressed;
                }
            }
        }
        311 | 621 => {
            for (_, mut interaction) in &mut buttons {
                *interaction = Interaction::None;
            }
        }
        360 | 480 => keys.press(KeyCode::Tab),
        361 | 481 => keys.release(KeyCode::Tab),
        391 => {
            for key in [
                KeyCode::KeyW,
                KeyCode::KeyD,
                KeyCode::KeyE,
                KeyCode::KeyR,
                KeyCode::KeyT,
                KeyCode::KeyY,
            ] {
                keys.press(key);
            }
        }
        440 => {
            for key in [
                KeyCode::KeyW,
                KeyCode::KeyD,
                KeyCode::KeyE,
                KeyCode::KeyR,
                KeyCode::KeyT,
                KeyCode::KeyY,
            ] {
                keys.release(key);
            }
        }
        530 | 570 => keys.press(KeyCode::Escape),
        531 | 571 => keys.release(KeyCode::Escape),
        710 => keys.press(KeyCode::F5),
        711 => keys.release(KeyCode::F5),
        780 => keys.press(KeyCode::F9),
        781 => keys.release(KeyCode::F9),
        _ => {}
    }
}
pub fn observe(
    mut check: ResMut<Check>,
    game: Res<Game>,
    mut validation: ResMut<Validation>,
    mut exit: MessageWriter<AppExit>,
    mut commands: Commands,
    windows: Query<(&Window, &CursorOptions), With<PrimaryWindow>>,
) {
    if check.path.is_none() || check.finished {
        return;
    }
    if check.screen_mode {
        observe_screen(&mut check, &game, &mut validation, &mut commands, &mut exit);
        return;
    }
    let s = &game.session;
    let position = crate::app::bvec(s.player.position);
    match check.frame {
        180 => {
            check.yaw = s.player.yaw;
            check.position = position;
        }
        181 => check.expect(
            game.mouse_captured
                && windows
                    .single()
                    .is_ok_and(|(_, c)| c.visible && c.grab_mode == CursorGrabMode::None),
            "click requests mouse look while automated mode leaves desktop cursor free",
            &mut validation,
        ),
        182 => {
            let yaw = check.yaw;
            check.expect(
                crate::app::bvec(s.look_direction()).dot(Vec3::new(-yaw.cos(), 0., yaw.sin()))
                    > 0.06,
                "mouse-right turns right immediately",
                &mut validation,
            );
        }
        215 => {
            let start = check.position;
            check.expect(
                position.distance(start) > 0.015,
                "one-frame W tap reaches authoritative worker",
                &mut validation,
            );
            check.position = position;
        }
        235 => {
            let start = check.position;
            let right = *Transform::default()
                .looking_to(crate::app::bvec(s.look_direction()), Vec3::Y)
                .right();
            check.expect(
                (position - start).dot(right) > 0.015,
                "D strafes toward camera screen-right",
                &mut validation,
            );
            check.position = position;
        }
        250 => {
            check.expect(
                game.help && game.controls.walk == 0.0,
                "help blocks held movement",
                &mut validation,
            );
            let path = check.path.as_ref().unwrap().with_file_name("help.png");
            commands
                .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
                .observe(bevy::render::view::screenshot::save_to_disk(path));
        }
        260 => {
            let start = check.position;
            check.expect(
                position.distance(start) < 0.015,
                "help overlay does not move the player",
                &mut validation,
            );
        }
        350 => check.expect(
            !game.demo && s.sluice.placement.is_none(),
            "New shift toolbar starts fresh operation",
            &mut validation,
        ),
        390 => {
            check.expect(
                s.player.mode == ControlMode::Excavator,
                "Tab enters the excavator through input handler",
                &mut validation,
            );
            check.joints = [s.machine.boom, s.machine.stick, s.machine.curl];
            check.position = crate::app::bvec(s.machine.position);
        }
        470 => {
            let joints = check.joints;
            let start = check.position;
            check.expect(
                s.machine.boom - joints[0] > 0.05
                    && s.machine.stick - joints[1] > 0.05
                    && s.machine.curl - joints[2] > 0.05,
                "held boom/stick/curl keys operate machine joints",
                &mut validation,
            );
            check.expect(
                crate::app::bvec(s.machine.position).distance(start) > 0.10,
                "W/D drives and turns tracks",
                &mut validation,
            );
        }
        520 => check.expect(
            s.player.mode == ControlMode::OnFoot,
            "Tab exits onto walkable ground",
            &mut validation,
        ),
        545 => {
            check.expect(s.paused, "Escape pauses with free cursor", &mut validation);
            check.clock = s.sim_time;
        }
        560 => {
            let clock = check.clock;
            check.expect(
                (s.sim_time - clock).abs() < 1e-8,
                "pause stops authoritative simulation",
                &mut validation,
            );
        }
        600 => {
            let clock = check.clock;
            check.expect(
                !s.paused && s.sim_time > clock,
                "Escape resumes simulation",
                &mut validation,
            );
        }
        700 => check.expect(
            game.demo
                && s.sluice.pump_on
                && s.sluice.feed_on
                && s.sluice.water.len() > 10_000
                && s.sluice.granular_packets
                && s.sluice.hopper.volume() > 0.4,
            "Demo toolbar restores already-flowing finite paydirt",
            &mut validation,
        ),
        760 => check.expect(
            s.message.starts_with("Shift saved"),
            "F5 saves live flowing demo",
            &mut validation,
        ),
        780 => validation.restore_requested = true,
        850 => {
            check.expect(
                validation.saved && s.sluice.pump_on && s.sluice.feed_on,
                "F9 restores the running particle simulation",
                &mut validation,
            );
            check.expect(
                (s.accounted_gold() - s.initial_gold_g).abs() < 1e-7,
                "gold ledger remains balanced after controls and restore",
                &mut validation,
            );
            let receipt = serde_json::json!({"passed":validation.failure.is_none(),"input_source":"synthetic Bevy keyboard/mouse and toolbar interactions in native app; OS focus and key delivery are not exercised", "checks":check.checks,"water_particles":s.sluice.water.len(),"hopper_m3":s.sluice.hopper.volume(),"gold_error_g":s.accounted_gold()-s.initial_gold_g,"save_restored":validation.saved});
            let path = check.path.as_ref().unwrap();
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = std::fs::write(path, serde_json::to_vec_pretty(&receipt).unwrap()) {
                validation.failure = Some(e.to_string());
            }
            println!("NATIVE INPUT CHECK: {receipt}");
            check.finished = true;
            validation.input_complete = validation.failure.is_none();
            if !std::env::args().any(|a| a == "--capture") {
                exit.write(if validation.failure.is_none() {
                    AppExit::Success
                } else {
                    AppExit::error()
                });
            }
        }
        _ => {}
    }
}

fn observe_screen(
    check: &mut Check,
    game: &Game,
    validation: &mut Validation,
    commands: &mut Commands,
    exit: &mut MessageWriter<AppExit>,
) {
    let s = &game.session;
    let plant = s.plant.as_ref();
    if check.frame == 400 {
        validation.restore_requested = true;
        validation.saved = false;
    }
    let observed = matches!(check.frame, 200 | 240 | 280 | 320 | 380 | 440 | 500);
    if observed {
        let path = check.path.as_ref().unwrap();
        let name = path.file_stem().unwrap_or_default().to_string_lossy();
        let photo = path.with_file_name(format!("{name}-screen-{}.png", check.frame));
        if let Some(parent) = path.parent() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                validation.failure = Some(error.to_string());
            }
        }
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(photo.clone()));
        check.screen_stages.push(serde_json::json!({"frame":check.frame,"sim_time":s.sim_time,"snapshot_id":game.snapshot_id,"paused":s.paused,"motor_on":plant.map(|p|p.screen.motor_on),"aperture_m":plant.map(|p|p.screen.config.aperture_m),"plant_material":plant.map(|p|p.material()),"hand":s.player.hand,"message":s.message,"screenshot":photo}));
    }
    match check.frame {
        200 => {
            check.screen_before = plant.map(|p| (p.material(), s.player.hand));
            check.clock = s.sim_time;
            check.expect(
                s.paused
                    && plant.is_some_and(|p| {
                        p.screen.motor_on
                            && !p.material().is_empty()
                            && (p.screen.config.aperture_m - 0.003).abs() < 1e-12
                    }),
                "Escape pauses the loaded running 3 mm screen with real stock",
                validation,
            );
        }
        240 => check.expect(
            s.paused
                && plant.is_some_and(|p| !p.screen.motor_on)
                && (s.sim_time - check.clock).abs() < 1e-8,
            "K stops the screen while the physical clock stays paused",
            validation,
        ),
        280 => {
            let transferred = check.screen_before.is_some_and(|(stock, hand)| {
                s.player.hand.wet_mass_kg() > hand.wet_mass_kg()
                    && (s.player.hand.volume() - hand.volume() - stock.volume()).abs() < 1e-10
                    && (s.player.hand.gold_g - hand.gold_g - stock.gold_g).abs() < 1e-9
                    && (s.player.hand.pore_water_m3 - hand.pore_water_m3 - stock.pore_water_m3)
                        .abs()
                        < 1e-10
            });
            check.expect(
                plant.is_some_and(|p| p.material().is_empty()) && transferred,
                "L moves actual screen and bin stock into hand without loss",
                validation,
            );
        }
        320 => check.expect(
            plant.is_some_and(|p| {
                !p.screen.motor_on && (p.screen.config.aperture_m - 0.006).abs() < 1e-12
            }),
            "M changes the stopped empty deck from 3 to 6 mm",
            validation,
        ),
        380 => {
            let path = if game.demo && std::env::var_os("SLUICE_SAVE_PATH").is_none() {
                crate::app::save_path().with_file_name("flowing-demo.json")
            } else {
                crate::app::save_path()
            };
            match mining_sim::storage::load(&path) {
                Ok(saved) => {
                    check.screen_saved = Some(
                        serde_json::json!({"path":path,"aperture_m":saved.plant.as_ref().map(|p|p.screen.config.aperture_m),"motor_on":saved.plant.as_ref().map(|p|p.screen.motor_on),"plant_material":saved.plant.as_ref().map(|p|p.material()),"hand":saved.player.hand}),
                    );
                    check.expect(
                        s.message.starts_with("Shift saved")
                            && saved.plant.as_ref().is_some_and(|p| {
                                !p.screen.motor_on
                                    && p.material().is_empty()
                                    && (p.screen.config.aperture_m - 0.006).abs() < 1e-12
                            })
                            && saved.player.hand == s.player.hand,
                        "F5 writes the actual modified deck and transferred hand stock",
                        validation,
                    );
                }
                Err(error) => {
                    validation.failure = Some(format!("Cannot verify screen save: {error}"))
                }
            }
        }
        440 => {
            check.clock = s.sim_time;
            let same_hand = check
                .screen_saved
                .as_ref()
                .is_some_and(|saved| saved["hand"] == serde_json::json!(s.player.hand));
            check.expect(
                validation.saved
                    && same_hand
                    && plant.is_some_and(|p| {
                        !p.screen.motor_on && (p.screen.config.aperture_m - 0.006).abs() < 1e-12
                    }),
                "F9 acknowledges restoration of saved 6 mm deck and hand stock",
                validation,
            );
        }
        500 => {
            check.expect(
                !s.paused && s.sim_time > check.clock && plant.is_some_and(|p| p.screen.motor_on),
                "K restarts the restored screen and physical time advances",
                validation,
            );
            check.expect(
                (s.accounted_gold() - s.initial_gold_g).abs() < 1e-7
                    && s.water_balance_error_m3().abs() < 2e-5,
                "Actual gold and environmental water budgets close after maintenance",
                validation,
            );
        }
        560 => {
            let photos_written = check.screen_stages.len() == 7
                && check.screen_stages.iter().all(|stage| {
                    stage["screenshot"].as_str().is_some_and(|path| {
                        std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.len() > 0)
                    })
                });
            check.expect(
                photos_written,
                "All seven native stage screenshots reached disk",
                validation,
            );
            check.expect(
                (s.accounted_gold() - s.initial_gold_g).abs() < 1e-7
                    && s.water_balance_error_m3().abs() < 2e-5,
                "Gold and water remain balanced through the final screenshot write",
                validation,
            );
        }
        _ => {}
    }
    if validation.failure.is_some() || check.frame == 560 {
        let mut receipt = serde_json::json!({"passed":validation.failure.is_none(),"failure":validation.failure,"input_source":"synthetic Bevy keyboard events through the native gameplay input handler; OS key delivery, focus and manual operation are not exercised","checks":check.checks,"before":check.screen_before,"stages":check.screen_stages,"actual_save":check.screen_saved,"gold_error_g":s.accounted_gold()-s.initial_gold_g,"water_error_m3":s.water_balance_error_m3(),"save_restored":validation.saved});
        if let Err(error) = std::fs::write(
            check.path.as_ref().unwrap(),
            serde_json::to_vec_pretty(&receipt).unwrap(),
        ) {
            validation.failure = Some(format!("Cannot write screen replay receipt: {error}"));
            receipt["passed"] = serde_json::json!(false);
            receipt["failure"] = serde_json::json!(validation.failure);
        }
        println!("NATIVE SCREEN INPUT CHECK: {receipt}");
        check.finished = true;
        validation.input_complete = validation.failure.is_none();
        exit.write(if validation.input_complete {
            AppExit::Success
        } else {
            AppExit::error()
        });
    }
}

/// Opt-in physical replay, separate from the native keyboard/toolbar checks.
/// It never writes a machine pose, soil body, material inventory or solver state.
pub fn soil_stroke(check: Res<Check>, mut game: ResMut<Game>) {
    if check.soil_motion && game.session.player.mode == ControlMode::Excavator {
        let input = mining_sim::replay::soil_trial_controls(&game.session, game.session.sim_time);
        game.latch_controls(mining_sim::session::Controls {
            machine: input,
            ..Default::default()
        });
    }
}
