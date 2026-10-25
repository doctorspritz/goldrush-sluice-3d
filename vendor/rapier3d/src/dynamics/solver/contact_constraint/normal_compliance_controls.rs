//! Actual scalar row versus an independent implicit spring/dashpot solution.
//! These are uncapped, fixed-normal controls, not a granular material model.
use super::*;
use crate::dynamics::SpringCoefficients;
use crate::dynamics::solver::SolverVel;
use crate::math::Vector;

#[test]
fn soft_normal_row_matches_implicit_spring_and_its_discrete_energy_identity() {
    let mut cases = 0;
    let mut worst_relative_balance = 0f64;
    for mass in [0.1f32, 2.5, 250.] {
        for frequency in [15f32, 30., 60.] {
            for damping in [0f32, 1., 10.] {
                for dt in [1f32 / 480., 1. / 1920.] {
                    for approach in [0f32, -0.2] {
                        let softness = SpringCoefficients::new(frequency, damping);
                        let gap = -0.004f32;
                        let mut row = ContactConstraintNormalPart::<Real> {
                            r: mass,
                            rhs: softness.erp_inv_dt(dt) * gap,
                            cfm_factor: softness.cfm_factor(dt),
                            ..ContactConstraintNormalPart::zero()
                        };
                        let mut a = SolverVel::<Real>::default();
                        a.linear.x = approach;
                        let mut b = SolverVel::<Real>::default();
                        row.solve(
                            &Vector::X,
                            &Vector::splat(mass.recip()),
                            &Vector::ZERO,
                            &mut a,
                            &mut b,
                        );
                        let m = f64::from(mass);
                        let h = f64::from(dt);
                        let x = f64::from(gap);
                        let v0 = f64::from(approach);
                        let omega = std::f64::consts::TAU * f64::from(frequency);
                        let k = m * omega * omega;
                        let d = 2. * f64::from(damping) * m * omega;
                        let expected = (m * v0 - h * k * x) / (m + h * d + h * h * k);
                        let v = f64::from(a.linear.x);
                        assert!(
                            (v - expected).abs() < 2e-7,
                            "m={m}, f={frequency}, z={damping}, h={h}: actual={v}, expected={expected}"
                        );
                        assert!(
                            (v - v0 - f64::from(row.impulse) / m).abs() < 2e-7,
                            "r is effective mass, not inverse mass"
                        );
                        let x1 = x + h * v;
                        assert!(
                            x1 < 0. && row.impulse > 0.,
                            "admitted spring must stay active"
                        );
                        let dk = 0.5 * m * (v * v - v0 * v0);
                        let du = 0.5 * k * (x1 * x1 - x * x);
                        let dashpot = h * d * v * v;
                        let numerical = 0.5 * m * (v - v0).powi(2) + 0.5 * k * (x1 - x).powi(2);
                        let scale = dk.abs() + du.abs() + dashpot + numerical;
                        let relative = (dk + du + dashpot + numerical).abs() / scale.max(1e-15);
                        worst_relative_balance = worst_relative_balance.max(relative);
                        assert!(
                            relative < 0.00002,
                            "discrete spring energy does not close: {relative}"
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    std::println!(
        "actual normal compliance: {cases} implicit-spring cases; worst relative energy closure={worst_relative_balance}"
    );
}

#[test]
fn compressed_contact_can_release_strain_without_creating_total_spring_energy() {
    let mass = 250f32;
    let dt = 1f32 / 1920.;
    let softness = SpringCoefficients::new(30., 10.);
    let mut row = ContactConstraintNormalPart::<Real> {
        r: mass,
        rhs: softness.erp_inv_dt(dt) * -0.004,
        cfm_factor: softness.cfm_factor(dt),
        ..ContactConstraintNormalPart::zero()
    };
    let mut a = SolverVel::<Real>::default();
    row.solve(
        &Vector::X,
        &Vector::splat(mass.recip()),
        &Vector::ZERO,
        &mut a,
        &mut SolverVel::default(),
    );
    let v = f64::from(a.linear.x);
    let h = f64::from(dt);
    let k = f64::from(mass) * (std::f64::consts::TAU * 30.).powi(2);
    let kinetic = 0.5 * f64::from(mass) * v * v;
    let strain_change = 0.5 * k * ((-0.004 + h * v).powi(2) - 0.004f64.powi(2));
    assert!(
        kinetic > 0.01,
        "negative control must expose mechanical-only positive work"
    );
    assert!(
        kinetic + strain_change < 0.,
        "the independent spring must dissipate total energy"
    );
    std::println!(
        "compressed scalar contact: mechanical gain={kinetic}J, contact strain change={strain_change}J; not a full-bank acceptance result"
    );
}

#[test]
fn near_rigid_frequency_is_not_a_well_conditioned_physical_spring_parameter() {
    let h = 1f32 / 1920.;
    let settings = SpringCoefficients::new(1.0e6, 1.);
    let factor = settings.cfm_factor(h);
    let gain = settings.erp_inv_dt(h);
    let gamma = f64::from(factor).recip() - 1.;
    let equivalent_k = f64::from(gain) / (f64::from(h) * gamma);
    let nominal_k = (std::f64::consts::TAU * 1.0e6).powi(2);
    let difference = (equivalent_k / nominal_k - 1.).abs();
    let steps = 1f32.to_bits() - factor.to_bits();
    assert_eq!(
        steps, 2,
        "reassess conditioning when float computation changes"
    );
    assert!((0.2..0.23).contains(&difference));
    std::println!(
        "near-rigid normal softness: factor={factor}, steps below one={steps}, equivalent vs nominal stiffness difference={difference}; numerical regularization is not calibrated material strain"
    );
}

#[test]
fn opening_soft_normal_contact_applies_no_tensile_dashpot_impulse() {
    let mass = 2.5f32;
    let h = 1f32 / 1920.;
    let softness = SpringCoefficients::new(30., 10.);
    let mut row = ContactConstraintNormalPart::<Real> {
        r: mass,
        rhs: softness.erp_inv_dt(h) * -0.001,
        cfm_factor: softness.cfm_factor(h),
        ..ContactConstraintNormalPart::zero()
    };
    let mut a = SolverVel::<Real>::default();
    a.linear.x = 1.;
    row.solve(
        &Vector::X,
        &Vector::splat(mass.recip()),
        &Vector::ZERO,
        &mut a,
        &mut SolverVel::default(),
    );
    assert_eq!(row.impulse, 0.);
    assert_eq!(a.linear.x, 1.);
}
