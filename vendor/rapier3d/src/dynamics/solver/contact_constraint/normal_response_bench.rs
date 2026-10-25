//! Actual pipeline response against an independently integrated specified spring.
//! Manufactured k/c controls, not a soil modulus or a Hertz contact calibration.
use crate::dynamics::{RigidBodyBuilder, SpringCoefficients};
use crate::geometry::ColliderBuilder;
use crate::math::Vector;
use crate::pipeline::PhysicsWorld;

#[derive(Clone, Copy)]
struct State {
    gap: f64,
    velocity: f64,
}
fn implicit_step(s: State, m: f64, k: f64, c: f64, h: f64, g: f64) -> State {
    let free = s.velocity + h * g;
    let velocity = (m * free - h * k * s.gap) / (m + h * c + h * h * k);
    let force = -k * (s.gap + h * velocity) - c * velocity;
    let velocity = if force > 0. { velocity } else { free };
    State {
        gap: s.gap + h * velocity,
        velocity,
    }
}
struct Measurement {
    position_error: f64,
    velocity_error: f64,
    final_gap: f64,
    final_velocity: f64,
    mass: f64,
    initial_energy: f64,
    maximum_energy: f64,
    analytic_position_error: f64,
    analytic_velocity_error: f64,
}
fn run(mass: f32, stiffness: f64, h: f32, loaded: bool, preserve: bool) -> Measurement {
    run_substeps(mass, stiffness, h, loaded, preserve, 1)
}
fn run_substeps(
    mass: f32,
    stiffness: f64,
    h: f32,
    loaded: bool,
    preserve: bool,
    substeps: usize,
) -> Measurement {
    let mut world = PhysicsWorld::new();
    let p = &mut world.integration_parameters;
    p.dt = h * substeps as f32;
    p.num_solver_iterations = substeps;
    p.num_internal_pgs_iterations = 8;
    assert_eq!(p.num_internal_stabilization_iterations, 1);
    p.diagnostic_preserve_normal_compliance = preserve;
    p.friction_in_bias_pass = true;
    p.friction_model = crate::dynamics::FrictionModel::Coulomb;
    let radius = 0.05f32;
    let density = mass / (4. * std::f32::consts::PI * radius.powi(3) / 3.);
    let gravity = if loaded { -9.81f32 } else { 0. };
    world.gravity = Vector::Y * gravity;
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::Y * -0.1),
        ColliderBuilder::cuboid(10., 0.1, 10.)
            .friction(0.)
            .restitution(0.),
    );
    let initial_gap = if loaded {
        -2. * f64::from(mass) * 9.81 / stiffness
    } else {
        -0.004
    };
    let (body, _) = world.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vector::Y * (radius + initial_gap as f32))
            .lock_rotations()
            .can_sleep(false),
        ColliderBuilder::ball(radius)
            .density(density)
            .friction(0.)
            .restitution(0.),
    );
    world.bodies[body].recompute_mass_properties_from_colliders(&world.colliders);
    let m = f64::from(world.bodies[body].mass());
    let frequency = (stiffness / m).sqrt() / std::f64::consts::TAU;
    let softness = SpringCoefficients::new(frequency as f32, 1.);
    world.integration_parameters.contact_softness = softness;
    world.integration_parameters.static_contact_softness = softness;
    let k = stiffness;
    let c = 2. * (k * m).sqrt();
    let h64 = f64::from(h);
    let g = f64::from(gravity);
    let mut reference = State {
        gap: f64::from(world.bodies[body].translation().y - radius),
        velocity: 0.,
    };
    let energy = |x: f64, v: f64| 0.5 * m * v * v + 0.5 * k * x.min(0.).powi(2) - m * g * x;
    let initial_energy = energy(reference.gap, 0.);
    let mut result = Measurement {
        position_error: 0.,
        velocity_error: 0.,
        final_gap: 0.,
        final_velocity: 0.,
        mass: m,
        initial_energy,
        maximum_energy: initial_energy,
        analytic_position_error: 0.,
        analytic_velocity_error: 0.,
    };
    let outer_h = h64 * substeps as f64;
    let steps = (0.5 / outer_h).ceil() as usize;
    let initial_state = reference;
    // Compare continuous error at a common 480 Hz observation cadence.
    // Independent grids can sample different transient maxima.
    let observation_stride = ((1. / 480.) / outer_h).round() as usize;
    for step in 0..steps {
        for _ in 0..substeps {
            reference = implicit_step(reference, m, k, c, h64, g);
        }
        world.step();
        let b = &world.bodies[body];
        let actual_gap = f64::from(b.translation().y - radius);
        let actual_velocity = f64::from(b.linvel().y);
        assert!(actual_gap.is_finite() && actual_velocity.is_finite());
        if loaded && (step + 1) % observation_stride == 0 {
            let t = (step + 1) as f64 * outer_h;
            let omega = (k / m).sqrt();
            let equilibrium = m * g / k;
            let amplitude = initial_state.gap - equilibrium;
            let exact_gap = equilibrium + amplitude * (1. + omega * t) * (-omega * t).exp();
            let exact_velocity = -amplitude * omega * omega * t * (-omega * t).exp();
            result.analytic_position_error = result
                .analytic_position_error
                .max((actual_gap - exact_gap).abs());
            result.analytic_velocity_error = result
                .analytic_velocity_error
                .max((actual_velocity - exact_velocity).abs());
        }
        result.position_error = result
            .position_error
            .max((actual_gap - reference.gap).abs());
        result.velocity_error = result
            .velocity_error
            .max((actual_velocity - reference.velocity).abs());
        result.maximum_energy = result
            .maximum_energy
            .max(energy(actual_gap, actual_velocity));
        result.final_gap = actual_gap;
        result.final_velocity = actual_velocity;
    }
    result
}
#[test]
fn complete_pipeline_preserves_a_specified_critical_normal_spring() {
    let mut count = 0;
    let mut pos = 0f64;
    let mut vel = 0f64;
    for mass in [1., 50.] {
        for stiffness in [100_000., 1_000_000.] {
            for h in [1. / 480., 1. / 1920.] {
                for loaded in [false, true] {
                    let m = run(mass, stiffness, h, loaded, true);
                    assert!(
                        m.position_error < 0.00003,
                        "m={mass} k={stiffness} h={h} loaded={loaded} position error {}",
                        m.position_error
                    );
                    assert!(
                        m.velocity_error < 0.002,
                        "m={mass} k={stiffness} h={h} loaded={loaded} velocity error {}",
                        m.velocity_error
                    );
                    assert!(
                        m.maximum_energy <= m.initial_energy + 0.0001,
                        "specified point spring gained total mechanical+strain energy"
                    );
                    if loaded {
                        let equilibrium = -m.mass * 9.81 / stiffness;
                        assert!(
                            (m.final_gap - equilibrium).abs() < 0.000003,
                            "load support differs from F/k"
                        );
                        assert!(
                            m.final_velocity.abs() < 0.0001,
                            "loaded contact failed to rest"
                        );
                    }
                    count += 1;
                    pos = pos.max(m.position_error);
                    vel = vel.max(m.velocity_error);
                }
            }
        }
    }
    std::println!(
        "NORMAL_RESPONSE {count} pipeline cases; maximum position error={pos}m, velocity error={vel}m/s"
    );
}
#[test]
fn ordinary_rigid_relaxation_is_not_a_physical_spring_transient() {
    let old = run(1., 250_000., 1. / 1920., false, false);
    let physical = run(1., 250_000., 1. / 1920., false, true);
    assert!(
        old.velocity_error > 0.1,
        "negative control missed erased spring response"
    );
    assert!(physical.velocity_error < 0.002);
    std::println!(
        "NORMAL_RELAXATION_CONTROL old velocity error={}m/s, preserved={}m/s",
        old.velocity_error,
        physical.velocity_error
    );
}

#[test]
fn supported_spring_response_converges_toward_the_continuous_solution() {
    for mass in [1., 50.] {
        for stiffness in [100_000., 1_000_000.] {
            let coarse = run(mass, stiffness, 1. / 480., true, true);
            let fine = run(mass, stiffness, 1. / 1920., true, true);
            assert!(
                fine.analytic_position_error < 0.65 * coarse.analytic_position_error,
                "position does not converge toward specified continuous spring"
            );
            assert!(
                fine.analytic_velocity_error < 0.65 * coarse.analytic_velocity_error,
                "velocity does not converge toward specified continuous spring"
            );
            std::println!(
                "NORMAL_REFINEMENT m={mass}kg k={stiffness}N/m position {}->{}m velocity {}->{}m/s",
                coarse.analytic_position_error,
                fine.analytic_position_error,
                coarse.analytic_velocity_error,
                fine.analytic_velocity_error
            );
        }
    }
}

#[test]
fn specified_normal_response_is_retained_through_four_temporal_substeps() {
    let mut count = 0;
    for mass in [1., 50.] {
        for loaded in [false, true] {
            let m = run_substeps(mass, 250_000., 1. / 1920., loaded, true, 4);
            assert!(
                m.position_error < 0.00003 && m.velocity_error < 0.002,
                "four-substep response differs from specified spring"
            );
            assert!(m.maximum_energy <= m.initial_energy + 0.0001);
            if loaded {
                assert!((m.final_gap + m.mass * 9.81 / 250_000.).abs() < 0.000003);
            }
            std::println!(
                "NORMAL_TEMPORAL m={mass} loaded={loaded} position_error={}m velocity_error={}m/s",
                m.position_error,
                m.velocity_error
            );
            count += 1;
        }
    }
    assert_eq!(count, 4);
}
