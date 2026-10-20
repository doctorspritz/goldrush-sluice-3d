use bevy::{
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    prelude::*,
    window::PrimaryWindow,
};
use mining_physics::worker::Command;
use mining_sim::{
    session::Controls, Action, ControlMode, MachineInput, Placement, Session, Vec3 as SimVec3,
};
use std::path::PathBuf;

#[derive(Resource)]
pub struct Game {
    pub session: Session,
    pub controls: Controls,
    pub placement: Option<Placement>,
    pub help: bool,
    pub pending: Vec<Command>,
    pub look_delta: [f32; 2],
    pub snapshot_id: u64,
    pub step_ms: f64,
    pub placement_error: Option<String>,
}
impl Default for Game {
    fn default() -> Self {
        Self {
            session: Session::new(),
            controls: Controls::default(),
            placement: None,
            help: false,
            pending: Vec::new(),
            look_delta: [0.0; 2],
            snapshot_id: 0,
            step_ms: 0.0,
            placement_error: None,
        }
    }
}
impl Game {
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
pub struct PlayerInput<'w> {
    keys: Res<'w, ButtonInput<KeyCode>>,
    mouse: Res<'w, ButtonInput<MouseButton>>,
    motion: Res<'w, AccumulatedMouseMotion>,
    scroll: Res<'w, AccumulatedMouseScroll>,
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
    } = input;
    if validation.automated() {
        return;
    }
    if keys.just_pressed(KeyCode::Escape) {
        if game.placement.is_some() {
            game.placement = None;
        } else {
            game.session.paused = !game.session.paused;
            let paused = game.session.paused;
            game.command(Command::Pause(paused));
        }
    }
    if keys.just_pressed(KeyCode::F1) {
        game.help = !game.help;
    }
    if keys.pressed(KeyCode::ControlLeft) && keys.just_pressed(KeyCode::KeyQ) {
        exit.write(AppExit::Success);
    }
    if keys.just_pressed(KeyCode::F5) {
        game.command(Command::Save(save_path()));
    }
    if keys.just_pressed(KeyCode::F9) {
        game.command(Command::Load(save_path()));
        game.placement = None;
    }
    if game.session.paused {
        game.controls = Controls::default();
        return;
    }
    if mouse.pressed(MouseButton::Right) {
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
            if let Some(cursor) = window.cursor_position() {
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
        if mouse.just_pressed(MouseButton::Left) {
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
        if mouse.just_pressed(MouseButton::Left) {
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
            if keys.just_pressed(KeyCode::ArrowUp) {
                game.command(Command::Water(0.002));
            }
            if keys.just_pressed(KeyCode::ArrowDown) {
                game.command(Command::Water(-0.002));
            }
            if keys.just_pressed(KeyCode::ArrowRight) {
                game.command(Command::Feed(0.001));
            }
            if keys.just_pressed(KeyCode::ArrowLeft) {
                game.command(Command::Feed(-0.001));
            }
        }
    }
    game.controls = Controls {
        walk: axis(KeyCode::KeyW, KeyCode::KeyS),
        strafe: axis(KeyCode::KeyD, KeyCode::KeyA),
        sprint: keys.pressed(KeyCode::ShiftLeft),
        machine: MachineInput {
            drive: axis(KeyCode::KeyW, KeyCode::KeyS),
            steer: axis(KeyCode::KeyD, KeyCode::KeyA),
            swing: axis(KeyCode::KeyE, KeyCode::KeyQ),
            boom: axis(KeyCode::KeyR, KeyCode::KeyF),
            stick: axis(KeyCode::KeyT, KeyCode::KeyG),
            curl: axis(KeyCode::KeyY, KeyCode::KeyH),
            cut: keys.pressed(KeyCode::Space),
            dump: keys.pressed(KeyCode::KeyJ),
        },
    };
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
            let right = Vec3::new(f.z, 0.0, -f.x);
            let p = bvec(s.machine.position) + Vec3::Y * 2.18 - right * 0.64 + f * 0.18;
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
