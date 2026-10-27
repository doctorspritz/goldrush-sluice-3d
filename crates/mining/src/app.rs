use bevy::{
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use mining_physics::worker::Command;
use mining_sim::{
    session::Controls, Action, ControlMode, MachineInput, Placement, Session, Vec3 as SimVec3,
};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FluidRevision {
    generation: u64,
    solver_steps: u64,
    outflow_time: u64,
    cleanups: u32,
    placement: Option<(SimVec3, f32, f32)>,
}

#[derive(Resource)]
pub struct Game {
    pub session: Session,
    pub controls: Controls,
    pub placement: Option<Placement>,
    pub demo: bool,
    pub help: bool,
    pub mouse_captured: bool,
    pub control_pulse: Controls,
    pub pending: Vec<Command>,
    pub look_delta: [f32; 2],
    pub snapshot_id: u64,
    pub simulation_generation: u64,
    pub step_ms: f64,
    pub placement_error: Option<String>,
}
impl Default for Game {
    fn default() -> Self {
        Self {
            session: Session::new_operation(),
            controls: Controls::default(),
            placement: None,
            demo: false,
            help: false,
            mouse_captured: false,
            control_pulse: Controls::default(),
            pending: Vec::new(),
            look_delta: [0.0; 2],
            snapshot_id: 0,
            simulation_generation: 0,
            step_ms: 0.0,
            placement_error: None,
        }
    }
}
impl Game {
    pub(crate) fn fluid_revision(&self) -> FluidRevision {
        let s = &self.session.sluice;
        FluidRevision {
            generation: self.simulation_generation,
            solver_steps: s.solver_steps,
            outflow_time: if s.outflow.is_empty() || !s.water.is_empty() {
                0
            } else {
                self.session.sim_time.to_bits()
            },
            cleanups: s.cleanups,
            placement: s.placement.map(|p| (p.inlet, p.yaw, p.slope_deg)),
        }
    }
    pub fn restart(&mut self, demo: bool) {
        self.stop_controls();
        self.pending.clear();
        self.look_delta = [0.0; 2];
        self.mouse_captured = false;
        self.help = false;
        self.placement = None;
        self.demo = demo;
        self.session = if demo {
            crate::demo::session().expect("Validated demo")
        } else {
            Session::new_operation()
        };
        self.snapshot_id += 1;
        self.command(Command::Reset(Box::new(self.session.clone())));
    }
    pub fn stop_controls(&mut self) {
        self.controls = Controls::default();
        self.control_pulse = Controls::default();
    }
    pub fn latch_controls(&mut self, controls: Controls) {
        let axis = |current: f32, pending: &mut f32| {
            if current != 0.0 {
                *pending = current;
            }
        };
        let p = &mut self.control_pulse;
        axis(controls.walk, &mut p.walk);
        axis(controls.strafe, &mut p.strafe);
        p.sprint |= controls.sprint;
        axis(controls.machine.drive, &mut p.machine.drive);
        axis(controls.machine.steer, &mut p.machine.steer);
        axis(controls.machine.swing, &mut p.machine.swing);
        axis(controls.machine.boom, &mut p.machine.boom);
        axis(controls.machine.stick, &mut p.machine.stick);
        axis(controls.machine.curl, &mut p.machine.curl);
        p.machine.cut |= controls.machine.cut;
        p.machine.dump |= controls.machine.dump;
        self.controls = controls;
    }
    pub fn take_controls(&mut self) -> Controls {
        let pulse = std::mem::take(&mut self.control_pulse);
        let choose = |current, pending| if current != 0.0 { current } else { pending };
        let c = self.controls;
        Controls {
            walk: choose(c.walk, pulse.walk),
            strafe: choose(c.strafe, pulse.strafe),
            sprint: c.sprint || pulse.sprint,
            machine: MachineInput {
                drive: choose(c.machine.drive, pulse.machine.drive),
                steer: choose(c.machine.steer, pulse.machine.steer),
                swing: choose(c.machine.swing, pulse.machine.swing),
                boom: choose(c.machine.boom, pulse.machine.boom),
                stick: choose(c.machine.stick, pulse.machine.stick),
                curl: choose(c.machine.curl, pulse.machine.curl),
                cut: c.machine.cut || pulse.machine.cut,
                dump: c.machine.dump || pulse.machine.dump,
            },
        }
    }
    pub fn command(&mut self, command: Command) {
        if self.pending.len() < 256 {
            self.pending.push(command);
        } else {
            self.session.message = "Simulation is busy; wait before issuing more commands".into();
        }
    }
    pub fn action(&mut self, action: Action) -> Result<(), String> {
        if self.pending.len() >= 256 {
            return Err("Simulation is busy".into());
        }
        if let Action::Place(p) = action {
            self.session.validate_placement(p)?;
        }
        self.command(Command::Action(action));
        Ok(())
    }
}
pub fn bvec(v: SimVec3) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}
pub fn svec(v: Vec3) -> SimVec3 {
    SimVec3::new(v.x, v.y, v.z)
}
#[derive(Component)]
pub struct GameCamera;

#[derive(bevy::ecs::system::SystemParam)]
pub struct PlayerInput<'w, 's> {
    keys: Res<'w, ButtonInput<KeyCode>>,
    mouse: Res<'w, ButtonInput<MouseButton>>,
    motion: Res<'w, AccumulatedMouseMotion>,
    scroll: Res<'w, AccumulatedMouseScroll>,
    ui: Query<'w, 's, &'static Interaction, With<Button>>,
}

pub fn input(
    input: PlayerInput,
    mut game: ResMut<Game>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<GameCamera>>,
    mut exit: MessageWriter<AppExit>,
    validation: Res<crate::validation::Validation>,
) {
    let PlayerInput {
        keys,
        mouse,
        motion,
        scroll,
        ui,
    } = input;
    if validation.automated() && !validation.input_check {
        return;
    }
    // Synthetic native checks feed only their own ECS inputs. They run without
    // desktop focus; normal play always keeps the actual window-focus gate.
    if !validation.input_check && !windows.single().is_ok_and(|w| w.focused) {
        game.mouse_captured = false;
        game.stop_controls();
        return;
    }
    if keys.just_pressed(KeyCode::Escape) {
        game.stop_controls();
        if game.placement.take().is_some() {
            game.mouse_captured = false;
        } else if game.help {
            game.help = false;
        } else if game.mouse_captured {
            game.mouse_captured = false;
        } else {
            game.session.paused = !game.session.paused;
            let paused = game.session.paused;
            game.command(Command::Pause(paused));
        }
    }
    if keys.just_pressed(KeyCode::F1) {
        game.help = !game.help;
        game.mouse_captured = false;
    }
    if keys.pressed(KeyCode::ControlLeft) && keys.just_pressed(KeyCode::KeyQ) {
        exit.write(AppExit::Success);
    }
    if keys.just_pressed(KeyCode::F5) {
        let path = active_save_path(game.demo);
        game.command(Command::Save(path));
    }
    if keys.just_pressed(KeyCode::F9) {
        let path = active_save_path(game.demo);
        game.command(Command::Load(path));
        game.placement = None;
    }
    if game.session.paused || game.help {
        game.stop_controls();
        return;
    }
    let over_ui = ui.iter().any(|i| *i != Interaction::None);
    let capture_click = !game.mouse_captured
        && game.placement.is_none()
        && mouse.just_pressed(MouseButton::Left)
        && !over_ui;
    if capture_click {
        game.mouse_captured = true;
    }
    if (game.mouse_captured || mouse.pressed(MouseButton::Right)) && !over_ui {
        let yaw = -motion.delta.x * 0.004;
        let pitch = (game.session.player.pitch - motion.delta.y * 0.003).clamp(-1.30, 1.10)
            - game.session.player.pitch;
        game.session.player.yaw += yaw;
        game.session.player.pitch += pitch;
        game.look_delta[0] += yaw;
        game.look_delta[1] += pitch;
    }
    let axis=|plus:KeyCode,minus:KeyCode| if keys.pressed(plus){1.0}else{0.0}-if keys.pressed(minus){1.0}else{0.0};
    let foot = game.session.player.mode == ControlMode::OnFoot;
    if keys.just_pressed(KeyCode::Tab) || (foot && keys.just_pressed(KeyCode::KeyE)) {
        let _ = game.action(Action::EnterExit);
    }
    if keys.just_pressed(KeyCode::KeyV) {
        let _ = game.action(Action::Camera);
    }
    if keys.just_pressed(KeyCode::KeyI) {
        let _ = game.action(Action::Engine);
    }
    if keys.just_pressed(KeyCode::KeyB) && foot {
        if game.placement.is_some() {
            game.placement = None;
        } else {
            game.mouse_captured = false;
            game.placement = Some(
                game.session
                    .sluice
                    .placement
                    .unwrap_or_else(|| Placement::suggested(&game.session.site)),
            );
        }
    }
    if let Some(mut p) = game.placement {
        if keys.just_pressed(KeyCode::KeyR) {
            p.yaw += 15.0f32.to_radians();
        }
        if keys.just_pressed(KeyCode::KeyQ) {
            p.yaw -= 15.0f32.to_radians();
        }
        if keys.just_pressed(KeyCode::BracketLeft) {
            p.slope_deg = (p.slope_deg - 0.5).max(3.0);
        }
        if keys.just_pressed(KeyCode::BracketRight) {
            p.slope_deg = (p.slope_deg + 0.5).min(12.0);
        }
        if scroll.delta.y != 0.0 {
            p.slope_deg = (p.slope_deg + scroll.delta.y.signum() * 0.5).clamp(3.0, 12.0);
        }
        if let (Ok(window), Ok((camera, transform))) = (windows.single(), camera.single()) {
            if let Some(cursor) = window
                .cursor_position()
                .filter(|_| motion.delta != Vec2::ZERO)
            {
                if let Ok(ray) = camera.viewport_to_world(transform, cursor) {
                    if let Some(hit) =
                        game.session
                            .site
                            .raycast(svec(ray.origin), svec(*ray.direction), 25.0)
                    {
                        let port = Placement::suggested(&game.session.site).inlet;
                        let snapped = SimVec3::new(hit.x, 0.0, hit.z)
                            .distance(SimVec3::new(port.x, 0.0, port.z))
                            < 2.0;
                        p.inlet = if snapped {
                            port
                        } else {
                            SimVec3::new(
                                hit.x,
                                game.session.site.height(hit.x, hit.z) + 0.85,
                                hit.z,
                            )
                        };
                    }
                }
            }
        }
        game.placement_error = game.session.validate_placement(p).err();
        if mouse.just_pressed(MouseButton::Left) && !over_ui {
            match game.action(Action::Place(p)) {
                Ok(()) => {
                    game.placement = None;
                    game.session.message =
                        "Sluice installed. P starts the pump; O starts the feed.".into();
                }
                Err(e) => {
                    game.session.message = e;
                    game.placement = Some(p);
                }
            }
        } else {
            game.placement = Some(p);
        }
    } else if foot {
        if mouse.just_pressed(MouseButton::Left) && !capture_click && !over_ui {
            let origin = game.session.player.position + SimVec3::Y * 1.5;
            if let Some(p) = game
                .session
                .site
                .raycast(origin, game.session.look_direction(), 3.0)
            {
                let _ = game.action(Action::Dig(p));
            } else {
                game.session.message = "Look down at reachable ground to use the shovel".into();
            }
        }
        if keys.just_pressed(KeyCode::KeyJ) {
            let _ = game.action(Action::DumpHand);
        }
        for (key, action) in [
            (KeyCode::KeyP, Action::TogglePump),
            (KeyCode::KeyO, Action::ToggleFeed),
            (KeyCode::KeyC, Action::Cleanup),
        ] {
            if keys.just_pressed(key) {
                let _ = game.action(action);
            }
        }
        if game.session.near_plant() {
            let coarse = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
            let water_step = if coarse { 0.002 } else { 0.00025 };
            let feed_step = if coarse { 0.0001 } else { 0.000025 };
            if keys.just_pressed(KeyCode::ArrowUp) {
                game.command(Command::Water(water_step));
            }
            if keys.just_pressed(KeyCode::ArrowDown) {
                game.command(Command::Water(-water_step));
            }
            if keys.just_pressed(KeyCode::ArrowRight) {
                game.command(Command::Feed(feed_step));
            }
            if keys.just_pressed(KeyCode::ArrowLeft) {
                game.command(Command::Feed(-feed_step));
            }
        }
    }
    game.latch_controls(Controls {
        walk: axis(KeyCode::KeyW, KeyCode::KeyS),
        strafe: axis(KeyCode::KeyD, KeyCode::KeyA),
        sprint: keys.pressed(KeyCode::ShiftLeft),
        machine: if foot {
            MachineInput::default()
        } else {
            MachineInput {
                drive: axis(KeyCode::KeyW, KeyCode::KeyS),
                steer: axis(KeyCode::KeyA, KeyCode::KeyD),
                swing: axis(KeyCode::KeyQ, KeyCode::KeyE),
                boom: axis(KeyCode::KeyR, KeyCode::KeyF),
                stick: axis(KeyCode::KeyT, KeyCode::KeyG),
                curl: axis(KeyCode::KeyY, KeyCode::KeyH),
                cut: keys.pressed(KeyCode::Space)
                    || (mouse.pressed(MouseButton::Left) && !capture_click && !over_ui),
                dump: keys.pressed(KeyCode::KeyJ),
            }
        },
    });
}
pub fn cursor(
    mut windows: Query<(&Window, &mut CursorOptions), With<PrimaryWindow>>,
    game: Res<Game>,
    validation: Res<crate::validation::Validation>,
) {
    for (window, mut cursor) in &mut windows {
        let capture = game.mouse_captured
            && window.focused
            && !game.session.paused
            && !game.help
            && game.placement.is_none()
            && !validation.automated();
        cursor.grab_mode = if capture {
            CursorGrabMode::Locked
        } else {
            CursorGrabMode::None
        };
        cursor.visible = !capture;
    }
}

pub fn camera(
    mut camera: Query<&mut Transform, With<GameCamera>>,
    game: Res<Game>,
    time: Res<Time>,
) {
    let Ok(mut camera) = camera.single_mut() else {
        return;
    };
    let s = &game.session;
    let look = bvec(s.look_direction());
    let (position, target) = if s.player.mode == ControlMode::Excavator {
        if s.player.third_person {
            let target = bvec(s.machine.position) + Vec3::Y * 1.5;
            (target - look * 9.0 + Vec3::Y * 4.5, target + look * 2.0)
        } else {
            let f = bvec(s.machine.arm().forward);
            let right = bvec(s.machine.arm().right);
            let p =
                bvec(s.machine.position) + bvec(s.machine.up()) * 2.18 - right * 0.64 + f * 0.18;
            (p, p + look * 5.0)
        }
    } else if s.player.third_person {
        let target = bvec(s.player.position) + Vec3::Y * 1.2;
        (target - look * 5.0 + Vec3::Y * 2.0, target + look * 1.8)
    } else {
        let p = bvec(s.player.position) + Vec3::Y * 1.68;
        (p, p + look * 5.0)
    };
    let ground = s.site.height(position.x, position.z) + 0.25;
    let position = Vec3::new(position.x, position.y.max(ground), position.z);
    let desired = Transform::from_translation(position).looking_at(target, Vec3::Y);
    let blend = 1.0 - (-time.delta_secs() * 14.0).exp();
    camera.translation = camera.translation.lerp(desired.translation, blend);
    camera.rotation = camera.rotation.slerp(desired.rotation, blend);
}
pub fn save_path() -> PathBuf {
    std::env::var_os("SLUICE_SAVE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir);
            if cfg!(target_os = "macos") {
                home.join("Library/Application Support/Sluice Moose/first-bend.json")
            } else if cfg!(target_os = "windows") {
                std::env::var_os("LOCALAPPDATA")
                    .map(PathBuf::from)
                    .unwrap_or(home)
                    .join("Sluice Moose/first-bend.json")
            } else {
                std::env::var_os("XDG_DATA_HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| home.join(".local/share"))
                    .join("sluice-moose/first-bend.json")
            }
        })
}

fn active_save_path(demo: bool) -> PathBuf {
    let path = save_path();
    if demo && std::env::var_os("SLUICE_SAVE_PATH").is_none() {
        path.with_file_name("flowing-demo.json")
    } else {
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::Validation;
    fn input_app() -> App {
        let mut app = App::new();
        app.init_resource::<Game>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .init_resource::<AccumulatedMouseMotion>()
            .init_resource::<AccumulatedMouseScroll>()
            .insert_resource(Validation::default())
            .add_message::<AppExit>()
            .add_systems(Update, (input, cursor).chain());
        app.world_mut().spawn((
            Window {
                focused: true,
                ..default()
            },
            PrimaryWindow,
        ));
        app
    }
    fn press(app: &mut App, key: KeyCode) {
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(key);
    }
    #[test]
    fn plant_rate_controls_have_fine_steps_and_shift_keeps_coarse_adjustment() {
        let mut app = input_app();
        {
            let mut game = app.world_mut().resource_mut::<Game>();
            game.session.player.position = SimVec3::new(16.3, 0.35, 20.);
            game.session.paused = false;
        }
        press(&mut app, KeyCode::ArrowUp);
        app.update();
        assert!(
            matches!(app.world().resource::<Game>().pending.last(),Some(Command::Water(v)) if (*v-0.00025).abs()<1e-9)
        );
        clear_edges(&mut app);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .release(KeyCode::ArrowUp);
        press(&mut app, KeyCode::ShiftLeft);
        press(&mut app, KeyCode::ArrowDown);
        app.update();
        assert!(
            matches!(app.world().resource::<Game>().pending.last(),Some(Command::Water(v)) if (*v+0.002).abs()<1e-9)
        );
    }
    fn clear_edges(app: &mut App) {
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .clear();
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = Vec2::ZERO;
    }
    #[test]
    fn real_input_path_captures_mouse_turns_right_and_escape_releases() {
        let mut app = input_app();
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        app.update();
        assert!(app.world().resource::<Game>().mouse_captured);
        {
            let cursor = app
                .world_mut()
                .query::<&CursorOptions>()
                .single(app.world())
                .unwrap();
            assert!(!cursor.visible && cursor.grab_mode == CursorGrabMode::Locked);
        }

        assert!(
            app.world().resource::<Game>().pending.is_empty(),
            "capture click must not dig"
        );
        clear_edges(&mut app);
        let right = *Transform::default()
            .looking_to(
                bvec(app.world().resource::<Game>().session.look_direction()),
                Vec3::Y,
            )
            .right();
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = Vec2::new(30.0, 10.0);
        app.update();
        let game = app.world().resource::<Game>();
        assert!(
            bvec(game.session.look_direction()).dot(right) > 0.10,
            "right mouse motion must turn toward camera screen-right"
        );
        assert!(game.session.player.pitch < -0.12);
        press(&mut app, KeyCode::Escape);
        app.update();
        assert!(!app.world().resource::<Game>().mouse_captured);
        assert!(!app.world().resource::<Game>().session.paused);
        let cursor = app
            .world_mut()
            .query::<&CursorOptions>()
            .single(app.world())
            .unwrap();
        assert!(cursor.visible && cursor.grab_mode == CursorGrabMode::None);
    }
    #[test]
    fn short_taps_survive_a_busy_worker_but_help_and_focus_loss_stop_input() {
        let mut app = input_app();
        press(&mut app, KeyCode::KeyW);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .release(KeyCode::KeyW);
        clear_edges(&mut app);
        app.update();
        assert_eq!(
            app.world_mut().resource_mut::<Game>().take_controls().walk,
            1.0
        );
        assert_eq!(
            app.world_mut().resource_mut::<Game>().take_controls().walk,
            0.0
        );
        press(&mut app, KeyCode::F1);
        press(&mut app, KeyCode::KeyW);
        app.update();
        assert!(app.world().resource::<Game>().help);
        assert_eq!(
            app.world_mut().resource_mut::<Game>().take_controls().walk,
            0.0
        );
        clear_edges(&mut app);
        app.world_mut().resource_mut::<Game>().help = false;
        let mut query = app.world_mut().query::<&mut Window>();
        query.single_mut(app.world_mut()).unwrap().focused = false;
        app.update();
        assert_eq!(
            app.world_mut().resource_mut::<Game>().take_controls().walk,
            0.0
        );
        assert!(!app.world().resource::<Game>().mouse_captured);
    }
    #[test]
    fn cab_keys_reach_the_machine_and_ui_clicks_do_not_mine() {
        let mut app = input_app();
        app.world_mut()
            .resource_mut::<Game>()
            .session
            .on_action(Action::EnterExit)
            .unwrap();
        for key in [
            KeyCode::KeyW,
            KeyCode::KeyD,
            KeyCode::KeyE,
            KeyCode::KeyR,
            KeyCode::KeyT,
            KeyCode::KeyY,
            KeyCode::Space,
            KeyCode::KeyJ,
        ] {
            press(&mut app, key);
        }
        app.update();
        let input = app
            .world_mut()
            .resource_mut::<Game>()
            .take_controls()
            .machine;
        assert_eq!(
            [
                input.drive,
                input.steer,
                input.swing,
                input.boom,
                input.stick,
                input.curl
            ],
            [1., -1., -1., 1., 1., 1.]
        );
        assert!(input.cut && input.dump);
        {
            let mut game = app.world_mut().resource_mut::<Game>();
            let right = *Transform::default()
                .looking_to(bvec(game.session.look_direction()), Vec3::Y)
                .right();
            game.session.step(Controls {
                machine: input,
                ..default()
            });
            assert!(
                bvec(game.session.machine.forward()).dot(right) > 0.,
                "D turns toward screen-right"
            );
            assert!(
                bvec(game.session.machine.arm().forward).dot(right) > 0.,
                "E swings toward screen-right"
            );
        }

        app.world_mut().spawn((Button, Interaction::Pressed));
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        app.update();
        assert!(!app.world().resource::<Game>().mouse_captured);
    }
}

#[cfg(test)]
mod screen_direction_tests {
    use super::*;
    #[test]
    fn d_strafes_toward_camera_right_across_headings() {
        for yaw in [0.0, 0.7, -1.57, 2.8] {
            let mut app = App::new();
            app.init_resource::<Game>()
                .init_resource::<ButtonInput<KeyCode>>()
                .init_resource::<ButtonInput<MouseButton>>()
                .init_resource::<AccumulatedMouseMotion>()
                .init_resource::<AccumulatedMouseScroll>()
                .init_resource::<crate::validation::Validation>()
                .add_message::<AppExit>()
                .add_systems(Update, input);
            app.world_mut().spawn((
                Window {
                    focused: true,
                    ..default()
                },
                PrimaryWindow,
            ));
            let (start, right) = {
                let mut game = app.world_mut().resource_mut::<Game>();
                game.session.player.yaw = yaw;
                game.session.player.position =
                    SimVec3::new(24., game.session.site.height(24., 16.), 16.);
                let start = game.session.player.position;
                let right = *Transform::default()
                    .looking_to(bvec(game.session.look_direction()), Vec3::Y)
                    .right();
                (start, right)
            };
            app.world_mut()
                .resource_mut::<ButtonInput<KeyCode>>()
                .press(KeyCode::KeyD);
            app.update();
            let mut game = app.world_mut().resource_mut::<Game>();
            let controls = game.take_controls();
            game.session.step(controls);
            assert!(
                bvec(game.session.player.position - start).dot(right) > 0.035,
                "yaw={yaw}: D must move right in the actual view"
            );
        }
    }
}

#[cfg(test)]
mod fluid_cache_tests {
    use super::*;
    #[test]
    fn airborne_only_ticks_do_not_rebuild_unchanged_channel_but_drain_stays_visible() {
        let mut game = Game::default();
        game.session.sluice.placement = Some(Placement::suggested(&game.session.site));
        game.session
            .sluice
            .outflow
            .push(mining_sim::outflow::OutflowParticle {
                id: 0,
                position: SimVec3::new(14., 2., 20.),
                velocity: SimVec3::X,
            });
        game.session
            .sluice
            .water
            .push(mining_sim::sluice::WaterParticle {
                id: 0,
                position: SimVec3::new(0.5, 0.4, 0.2),
                velocity: SimVec3::X,
                affine: Default::default(),
            });
        let key = game.fluid_revision();
        game.session.sim_time += mining_sim::FIXED_DT as f64;
        assert_eq!(key, game.fluid_revision());
        game.session.sluice.solver_steps += 4;
        assert_ne!(key, game.fluid_revision());
        game.session.sluice.water.clear();
        let key = game.fluid_revision();
        game.session.sim_time += mining_sim::FIXED_DT as f64;
        assert_ne!(
            key,
            game.fluid_revision(),
            "airborne water must keep moving after channel empties"
        );
        let key = game.fluid_revision();
        game.session.sluice.outflow.clear();
        assert_ne!(key, game.fluid_revision());
    }
}
