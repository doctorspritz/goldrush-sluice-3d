mod activity;
mod app;
mod control_check;
mod demo;
mod environment;
mod fluid_surface;
mod foliage;
mod ground_material;
mod hud;
mod motion_capture;
mod render_exit;
mod runtime;
mod scene;
mod sediment;
mod terrain_water;
mod validation;
mod water;
mod water_passes;
mod water_presentation;
mod water_surface_worker;

use bevy::{prelude::*, window::WindowResolution};

fn main() -> AppExit {
    // Keep Bevy's compute/IO pools bounded as well as the single solver worker.
    use bevy::app::{TaskPoolOptions, TaskPoolPlugin};
    let (mut session, demo) = demo::initial();
    if std::env::args().any(|a| a == "--payload-motion") {
        let args: Vec<_> = std::env::args().collect();
        assert!(
            ["--new-shift", "--articulated-machine", "--capture-motion"]
                .iter()
                .all(|s| args.iter().any(|a| a == s)),
            "payload probe requires a bounded new-shift articulated capture"
        );
        assert!(
            session.machine.bucket.is_empty(),
            "payload probe starts empty before finite fixture preparation"
        );
        let material = session
            .site
            .dig(mining_sim::Vec3::new(19., 0., 20.), 0.4, 0.2, 0.08);
        assert!(material.volume() > 0.01, "finite site fixture material");
        session
            .machine
            .enable_articulation()
            .expect("probe mechanism");
        let (w, q) = mining_sim::articulation::payload_frame(&session.machine);
        let parcel = mining_sim::payload::RigidParcel::cuboid(
            material,
            w + q * mining_sim::Vec3::new(0., -0.60, 0.40).as_dvec3(),
            mining_sim::Vec3::new(0.30, (material.volume() / (8. * 0.30 * 0.23)) as f32, 0.23)
                .as_dvec3(),
            q,
            Default::default(),
            Default::default(),
        )
        .expect("finite volume fixture");
        session.falling.push(parcel.into_falling());
        session
            .capture_rigid_load(0, 1.)
            .expect("finite initial fixture capture");
        session.message =
            "Finite spatial-load initial condition; this probe is not a scoop/dump replay.".into();
    }

    if std::env::args().any(|a| a == "--articulated-machine") {
        session
            .machine
            .enable_articulation()
            .expect("Articulated mechanism must initialize");
    }
    if std::env::args().any(|a| a == "--active-soil") {
        assert!(
            std::env::args().any(|a| a == "--new-shift"),
            "active-soil requires a fresh shift"
        );
        session
            .prepare_soil_trial()
            .expect("Finite soil trial must initialize");
    }
    session.sluice.ensure_water_ids();
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
        .add_plugins(render_exit::RenderExitPlugin)
        .insert_resource(game)
        .insert_non_send_resource(runtime)
        .insert_non_send_resource(activity::SimulationActivity::default())
        .init_resource::<validation::Validation>()
        .init_resource::<motion_capture::MotionCapture>()
        .init_resource::<environment::Detail>()
        .insert_resource(control_check::Check::from_args())
        .add_systems(Startup, (scene::setup, hud::setup))
        .add_systems(
            Update,
            (
                runtime::receive,
                control_check::inject,
                hud::buttons,
                (app::input, control_check::soil_stroke).chain(),
                app::cursor,
                control_check::observe,
                (app::camera, motion_capture::camera).chain(),
                environment::update_detail,
                (scene::sync_machine, scene::sync_soil).chain(),
                scene::sync_terrain,
                water::sync,
                scene::sync_plant,
                scene::sync_material,
                (sediment::sync, sediment::sync_tailings).chain(),
                scene::guides,
                hud::update,
                validation::update,
                motion_capture::update,
                activity::sync,
                runtime::dispatch,
            )
                .chain(),
        )
        .run()
}
