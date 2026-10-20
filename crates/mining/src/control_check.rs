//! Native regression through Bevy's keyboard/mouse and toolbar handlers. Inputs
//! are synthetic ECS events, not OS key injection or a manual playtest.
use crate::{app::Game, hud::HudButton, validation::Validation};
use bevy::{
    input::mouse::AccumulatedMouseMotion,
    prelude::*,
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
}
impl Check {
    pub fn from_args() -> Self {
        let args: Vec<_> = std::env::args().collect();
        Self {
            path: args
                .iter()
                .position(|a| a == "--check-controls")
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
    mut keys: ResMut<ButtonInput<KeyCode>>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    mut motion: ResMut<AccumulatedMouseMotion>,
    mut buttons: Query<(&HudButton, &mut Interaction)>,
) {
    if check.path.is_none() || check.finished {
        return;
    }
    check.frame += 1;
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
                && s.sluice.water.len() > 100_000
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
