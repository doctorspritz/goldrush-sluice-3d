//! Double-precision oracle for the locally implicit phase/pressure response.
//! Velocities passed here are body-force predictors. Pressure is an independent
//! impulse gradient: buoyancy must not be folded into the gravity predictor.
use glam::DVec3;

#[derive(Clone, Copy, Debug)]
pub struct GrainSample {
    pub volume: f64,
    pub density: f64,
    pub relaxation_rate: f64,
    pub velocity: DVec3,
}
impl GrainSample {
    pub fn mass(self) -> f64 {
        self.volume * self.density
    }
    pub fn relaxation(self, dt: f64) -> (f64, f64) {
        let a = self.relaxation_rate * dt;
        if a.is_infinite() {
            (1., 0.)
        } else {
            (a / (1. + a), 1. / (1. + a))
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FaceResponse {
    pub mixture_predictor: DVec3,
    pub pressure_mobility: f64,
    pub water_predictor: DVec3,
    pub water_pressure_mobility: f64,
    pub pressure_scale: f64,
    dt: f64,
}
impl FaceResponse {
    pub fn new(
        water_volume: f64,
        water_velocity: DVec3,
        grains: &[GrainSample],
        reference_volume: f64,
        dt: f64,
    ) -> Option<Self> {
        if !water_volume.is_finite()
            || water_volume <= 0.
            || !water_velocity.is_finite()
            || !reference_volume.is_finite()
            || reference_volume <= 0.
            || !dt.is_finite()
            || dt <= 0.
        {
            return None;
        }
        let (mut volume, mut b_volume, mut b_mass, mut b_momentum, mut r_volume, mut r_momentum) =
            (water_volume, 0., 0., DVec3::ZERO, 0., DVec3::ZERO);
        for &g in grains {
            if !g.volume.is_finite()
                || g.volume < 0.
                || !g.density.is_finite()
                || g.density <= 1.
                || !g.relaxation_rate.is_finite()
                || g.relaxation_rate < 0.
                || !g.velocity.is_finite()
            {
                return None;
            }
            let (b, c) = g.relaxation(dt);
            volume += g.volume;
            b_volume += g.volume * b;
            b_mass += g.mass() * b;
            b_momentum += g.velocity * (g.mass() * b);
            r_volume += g.volume * c / g.density;
            r_momentum += g.velocity * (g.volume * c);
        }
        let a = water_volume + b_volume;
        let m = water_volume + b_mass;
        let pressure_scale = reference_volume / volume;
        let water_predictor = (water_velocity * water_volume + b_momentum) / m;
        let mixture_predictor = (water_predictor * a + r_momentum) / volume;
        let pressure_mobility = pressure_scale / volume * (a * a / m + r_volume);
        let water_pressure_mobility = pressure_scale * a / m;
        if !mixture_predictor.is_finite()
            || !water_predictor.is_finite()
            || !pressure_mobility.is_finite()
            || pressure_mobility <= 0.
            || !water_pressure_mobility.is_finite()
            || !pressure_scale.is_finite()
        {
            return None;
        }
        Some(Self {
            mixture_predictor,
            pressure_mobility,
            water_predictor,
            water_pressure_mobility,
            pressure_scale,
            dt,
        })
    }

    /// `gradient` is minus the pressure gradient with the timestep absorbed.
    pub fn water(self, gradient: DVec3) -> DVec3 {
        self.water_predictor + self.water_pressure_mobility * gradient
    }
    pub fn grain(self, grain: GrainSample, gradient: DVec3) -> DVec3 {
        let (b, c) = grain.relaxation(self.dt);
        grain.velocity * c
            + self.water(gradient) * b
            + gradient * (c * self.pressure_scale / grain.density)
    }
    pub fn mixture(self, gradient: DVec3) -> DVec3 {
        self.mixture_predictor + self.pressure_mobility * gradient
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn samples() -> Vec<GrainSample> {
        vec![
            GrainSample {
                volume: 0.003,
                density: 2.65,
                relaxation_rate: 0.04,
                velocity: DVec3::new(-0.4, 0.1, 0.3),
            },
            GrainSample {
                volume: 16.,
                density: 19.3,
                relaxation_rate: 140.,
                velocity: DVec3::new(0.05, -0.2, 0.6),
            },
            GrainSample {
                volume: 0.2,
                density: 2.65,
                relaxation_rate: 8000.,
                velocity: DVec3::new(0.8, 0.6, -0.1),
            },
        ]
    }
    fn close(a: DVec3, b: DVec3, tol: f64) {
        assert!((a - b).length() < tol, "{a:?} versus {b:?}");
    }
    /// Independent dense equations, not the eliminated response formula.
    fn dense_solve(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
        let n = rhs.len();
        for i in 0..n {
            let pivot = (i..n)
                .max_by(|&a, &b| matrix[a][i].abs().total_cmp(&matrix[b][i].abs()))
                .unwrap();
            matrix.swap(i, pivot);
            rhs.swap(i, pivot);
            assert!(matrix[i][i].abs() > 1e-20);
            let scale = matrix[i][i];
            for value in &mut matrix[i][i..] {
                *value /= scale;
            }
            rhs[i] /= scale;
            let pivot_row = matrix[i].clone();
            for (j, row) in matrix.iter_mut().enumerate() {
                if j == i {
                    continue;
                }
                let factor = row[i];
                for (value, pivot) in row[i..].iter_mut().zip(&pivot_row[i..]) {
                    *value -= factor * pivot;
                }
                rhs[j] -= factor * rhs[i];
            }
        }
        rhs
    }
    #[test]
    fn eliminated_phase_response_matches_independent_dense_pressure_constraint() {
        let grains = samples();
        let water = DVec3::new(0.3, -0.1, 0.2);
        for dt in [1e-5, 1. / 240., 1. / 120., 0.1] {
            for vw in [0.1, 3., 100.] {
                let volume = vw + grains.iter().map(|g| g.volume).sum::<f64>();
                let face = FaceResponse::new(vw, water, &grains, 8., dt).unwrap();
                let target = DVec3::new(0.02, -0.3, 0.4);
                let gradient = (target - face.mixture_predictor) / face.pressure_mobility;
                for axis in 0..3 {
                    let n = grains.len() + 2;
                    let pressure = n - 1;
                    let mut matrix = vec![vec![0.; n]; n];
                    let mut rhs = vec![0.; n];
                    matrix[0][0] = vw;
                    rhs[0] = vw * water[axis];
                    for (i, g) in grains.iter().enumerate() {
                        let a = g.relaxation_rate * dt;
                        matrix[0][0] += g.mass() * a;
                        matrix[0][i + 1] = -g.mass() * a;
                        matrix[i + 1][0] = -a;
                        matrix[i + 1][i + 1] = 1. + a;
                        matrix[i + 1][pressure] = -8. / volume / g.density;
                        rhs[i + 1] = g.velocity[axis];
                        matrix[pressure][i + 1] = g.volume;
                    }
                    matrix[0][pressure] = -8. * vw / volume;
                    matrix[pressure][0] = vw;
                    rhs[pressure] = volume * target[axis];
                    let solution = dense_solve(matrix, rhs);
                    assert!((solution[0] - face.water(gradient)[axis]).abs() < 1e-8);
                    assert!((solution[pressure] - gradient[axis]).abs() < 1e-7);
                    for (i, g) in grains.iter().enumerate() {
                        assert!((solution[i + 1] - face.grain(*g, gradient)[axis]).abs() < 1e-8);
                    }
                }
            }
        }
    }
    #[test]
    fn drag_conserves_momentum_and_satisfies_backward_euler_dissipation_identity() {
        let grains = samples();
        let vw = 2.;
        let water = DVec3::new(-0.3, 0.4, -0.8);
        for dt in [1e-8, 1. / 240., 1. / 120., 1., 10000.] {
            let face = FaceResponse::new(vw, water, &grains, 8., dt).unwrap();
            let final_water = face.water(DVec3::ZERO);
            let old_p = water * vw + grains.iter().map(|g| g.velocity * g.mass()).sum::<DVec3>();
            let new_p = final_water * vw
                + grains
                    .iter()
                    .map(|g| face.grain(*g, DVec3::ZERO) * g.mass())
                    .sum::<DVec3>();
            close(old_p, new_p, 1e-11);
            let old_e = 0.5
                * (vw * water.length_squared()
                    + grains
                        .iter()
                        .map(|g| g.mass() * g.velocity.length_squared())
                        .sum::<f64>());
            let new_e = 0.5
                * (vw * final_water.length_squared()
                    + grains
                        .iter()
                        .map(|g| g.mass() * face.grain(*g, DVec3::ZERO).length_squared())
                        .sum::<f64>());
            let drag_loss = grains
                .iter()
                .map(|g| {
                    dt * g.mass()
                        * g.relaxation_rate
                        * (final_water - face.grain(*g, DVec3::ZERO)).length_squared()
                })
                .sum::<f64>();
            let increment_loss = 0.5
                * (vw * (final_water - water).length_squared()
                    + grains
                        .iter()
                        .map(|g| {
                            g.mass() * (face.grain(*g, DVec3::ZERO) - g.velocity).length_squared()
                        })
                        .sum::<f64>());
            assert!(new_e <= old_e + 1e-11);
            assert!(((old_e - new_e) - (drag_loss + increment_loss)).abs() < 1e-8);
        }
    }
    #[test]
    fn pressure_impulses_sum_to_the_reference_face_volume() {
        let grains = samples();
        let vw = 0.3;
        let water = DVec3::new(0.4, -0.7, 0.3);
        let gradient = DVec3::new(0.17, -0.06, 0.2);
        let face = FaceResponse::new(vw, water, &grains, 8., 1. / 120.).unwrap();
        let change = (face.water(gradient) - water) * vw
            + grains
                .iter()
                .map(|g| (face.grain(*g, gradient) - g.velocity) * g.mass())
                .sum::<DVec3>();
        close(change, gradient * 8., 1e-11);
        let volume = vw + grains.iter().map(|g| g.volume).sum::<f64>();
        let actual = (face.water(gradient) * vw
            + grains
                .iter()
                .map(|g| face.grain(*g, gradient) * g.volume)
                .sum::<DVec3>())
            / volume;
        close(actual, face.mixture(gradient), 1e-12);
    }
    #[test]
    fn stiff_limit_and_coincident_subdivision_retain_the_same_response() {
        let mut grains = samples();
        for g in &mut grains {
            g.relaxation_rate = 1e25;
        }
        let water = DVec3::new(-0.2, 0.1, 0.5);
        let gradient = DVec3::new(0.1, -0.2, 0.3);
        let face = FaceResponse::new(1., water, &grains, 8., 1. / 120.).unwrap();
        for g in &grains {
            close(face.grain(*g, gradient), face.water(gradient), 1e-12);
        }
        let split: Vec<_> = grains
            .iter()
            .flat_map(|g| {
                (0..16).map(move |_| GrainSample {
                    volume: g.volume / 16.,
                    ..*g
                })
            })
            .collect();
        let divided = FaceResponse::new(1., water, &split, 8., 1. / 120.).unwrap();
        close(face.water(gradient), divided.water(gradient), 1e-12);
        assert!((face.pressure_mobility - divided.pressure_mobility).abs() < 1e-14);
    }
    #[test]
    fn freefall_keeps_comoving_phases_together_for_every_density() {
        let predicted = DVec3::new(0.3, 0.2 - 9.81 / 120., -0.1);
        let mut grains = samples();
        for g in &mut grains {
            g.velocity = predicted;
        }
        let face = FaceResponse::new(1., predicted, &grains, 8., 1. / 120.).unwrap();
        close(face.water(DVec3::ZERO), predicted, 1e-12);
        for g in grains {
            close(face.grain(g, DVec3::ZERO), predicted, 1e-12);
        }
    }
    #[test]
    fn dilute_hydrostatic_pressure_produces_archimedes_settling_without_reduced_gravity() {
        for rho in [2.65, 19.3] {
            let (dt, gamma, g) = (1. / 120., 40., 9.81);
            let mut water = DVec3::ZERO;
            let mut grain = GrainSample {
                volume: 1e-10,
                density: rho,
                relaxation_rate: gamma,
                velocity: DVec3::ZERO,
            };
            for _ in 0..600 {
                water.y -= g * dt;
                grain.velocity.y -= g * dt;
                let face = FaceResponse::new(1., water, &[grain], 1. + grain.volume, dt).unwrap();
                let pressure = DVec3::Y * (g * dt);
                water = face.water(pressure);
                grain.velocity = face.grain(grain, pressure);
            }
            assert!(((grain.velocity - water).y + g * (1. - 1. / rho) / gamma).abs() < 1e-7);
        }
    }
    #[test]
    fn pure_water_and_invalid_dry_domains_are_explicit() {
        let v = DVec3::new(0.2, -0.3, 0.4);
        let face = FaceResponse::new(2., v, &[], 8., 0.01).unwrap();
        close(face.water_predictor, v, 1e-14);
        assert_eq!(face.pressure_mobility, 4.);
        assert_eq!(face.water_pressure_mobility, 4.);
        assert!(FaceResponse::new(0., v, &samples(), 8., 0.01).is_none());
        assert!(FaceResponse::new(2., v, &samples(), 8., f64::NAN).is_none());
    }
}
