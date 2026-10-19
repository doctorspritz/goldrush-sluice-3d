mod app;
mod hud;
mod scene;
mod validation;

use bevy::{prelude::*, window::WindowResolution};

fn main() -> AppExit {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "The Sluice Moose — First Bend".into(),
                resolution: WindowResolution::new(1440, 900).with_scale_factor_override(1.0),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(Time::<Fixed>::from_hz(60.0))
        .init_resource::<app::Game>()
        .insert_non_send_resource(
            mining_physics::Hydraulics::new()
                .expect("Original hydraulic simulation must initialize"),
        )
        .init_resource::<validation::Validation>()
        .add_systems(Startup, (scene::setup, hud::setup))
        .add_systems(FixedUpdate, validation::fixed)
        .add_systems(
            Update,
            (
                app::input,
                app::camera,
                scene::sync_machine,
                scene::sync_terrain,
                scene::sync_sluice_water,
                scene::sync_plant,
                scene::sync_material,
                scene::guides,
                hud::update,
                validation::update,
            )
                .chain(),
        )
        .run()
}
