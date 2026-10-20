mod app;
mod hud;
mod runtime;
mod scene;
mod validation;
mod water;

use bevy::{prelude::*, window::WindowResolution};

fn main() -> AppExit {
    // Keep Bevy's compute/IO pools bounded as well as the single solver worker.
    use bevy::app::{TaskPoolOptions, TaskPoolPlugin};
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
        .init_resource::<app::Game>()
        .insert_non_send_resource(runtime::Runtime::new().expect("Simulation worker must start"))
        .init_resource::<validation::Validation>()
        .add_systems(Startup, (scene::setup, hud::setup))
        .add_systems(
            Update,
            (
                runtime::receive,
                app::input,
                app::camera,
                scene::sync_machine,
                scene::sync_terrain,
                water::sync,
                scene::sync_plant,
                scene::sync_material,
                scene::guides,
                hud::update,
                validation::update,
                runtime::dispatch,
            )
                .chain(),
        )
        .run()
}
