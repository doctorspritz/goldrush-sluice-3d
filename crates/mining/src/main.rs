mod activity;
mod app;
mod control_check;
mod demo;
mod hud;
mod runtime;
mod scene;
mod sediment;
mod terrain_water;
mod validation;
mod water;

use bevy::{prelude::*, window::WindowResolution};

fn main() -> AppExit {
    // Keep Bevy's compute/IO pools bounded as well as the single solver worker.
    use bevy::app::{TaskPoolOptions, TaskPoolPlugin};
    let (session, demo) = demo::initial();
    let game = app::Game {
        session,
        demo,
        ..default()
    };
    let runtime =
        runtime::Runtime::new(game.session.clone()).expect("Simulation worker must start");
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(TaskPoolPlugin {
                    task_pool_options: TaskPoolOptions::with_num_threads(2),
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "The Sluice Moose — First Bend".into(),
                        resolution: WindowResolution::new(1440, 900)
                            .with_scale_factor_override(1.0),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(water::WaterSplatPlugin)
        .insert_resource(game)
        .insert_non_send_resource(runtime)
        .insert_non_send_resource(activity::SimulationActivity::default())
        .init_resource::<validation::Validation>()
        .insert_resource(control_check::Check::from_args())
        .add_systems(Startup, (scene::setup, hud::setup))
        .add_systems(
            Update,
            (
                runtime::receive,
                control_check::inject,
                hud::buttons,
                app::input,
                app::cursor,
                control_check::observe,
                app::camera,
                scene::sync_machine,
                scene::sync_terrain,
                water::sync,
                scene::sync_plant,
                scene::sync_material,
                sediment::sync,
                scene::guides,
                hud::update,
                validation::update,
                activity::sync,
                runtime::dispatch,
            )
                .chain(),
        )
        .run()
}
