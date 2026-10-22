//! Conservative f64 passive provenance for coarse paydirt. Replay the actual
//! water and bed events; a sum of face volumes cannot reproduce substep mixing.
//! Gold separation in the sluice remains the FLIP/DEM model.
use mining_sim::{site::WIDTH, Site};

pub(super) struct TerrainGold {
    mobile: Vec<f64>,
    local_precision: Vec<f64>,
}
impl TerrainGold {
    pub(super) fn new(site: &Site, depth: &[f32], area: f64) -> Self {
        let mut mobile = vec![0.; depth.len()];
        let mut local_precision = vec![0.; depth.len()];
        for (i, &h) in depth.iter().enumerate() {
            let gold = site
                .hydraulic
                .suspended_gold_g
                .get(i)
                .copied()
                .unwrap_or(0.);
            let represented = site
                .hydraulic
                .suspended_paydirt
                .get(i)
                .copied()
                .unwrap_or(0.) as f64
                * h as f64
                * area;
            let precision = site
                .hydraulic
                .suspended_remainder_m3
                .get(i)
                .map(|v| v[3])
                .unwrap_or(0.);
            let total = represented + precision;
            mobile[i] = if total > 0. {
                gold * (represented / total)
            } else {
                0.
            };
            local_precision[i] = gold - mobile[i];
        }
        Self {
            mobile,
            local_precision,
        }
    }
    pub(super) fn replay(
        mut self,
        site: &mut Site,
        events: &[[f32; 4]],
        substeps: usize,
        retained: &[[f64; 4]],
        area: f64,
        dt: f64,
    ) -> Result<(), String> {
        let n = WIDTH * WIDTH;
        if events.len() != (substeps + 2) * n || events.iter().flatten().any(|v| !v.is_finite()) {
            return Err("Invalid terrain provenance events".into());
        }
        let before_scour = &events[substeps * n..(substeps + 1) * n];
        let ws = (4. * 9.81 * 0.002 * 1650_f64 / (3. * 1000. * 0.44)).sqrt();
        for step in 0..substeps {
            let event = &events[step * n..(step + 1) * n];
            let next_h: Vec<f64> = (0..n)
                .map(|i| {
                    if step + 1 < substeps {
                        events[(step + 1) * n + i][0] as f64
                    } else {
                        before_scour[i][3] as f64
                    }
                })
                .collect();
            reactive_advection(
                &mut self.mobile,
                &mut site.gold_g,
                &mut site.hydraulic.exported_gold_g,
                event,
                &next_h,
                WIDTH,
                area,
                dt / substeps as f64,
                ws,
            )?;
        }
        // Reaction deposits already received their tags; only the subsequent
        // scour delta remains. No initial bed copies or second deposit replay.
        for event in events[(substeps + 1) * n..].chunks_exact(n) {
            for (i, record) in event.iter().enumerate() {
                if record.iter().any(|v| *v < 0.) {
                    return Err("Negative terrain bed provenance event".into());
                }
                // Subtract the two parts separately: adding a large geometric
                // height first would erase a tiny but counted bed transfer.
                let delta = (record[0] as f64 - before_scour[i][0] as f64)
                    + (record[1] as f64 - before_scour[i][1] as f64);
                exchange(
                    &mut site.gold_g[i],
                    &mut self.mobile[i],
                    before_scour[i][0] as f64 + before_scour[i][1] as f64 + retained[i][3] / area,
                    delta,
                    record[2] as f64 * record[3] as f64,
                );
            }
        }
        site.hydraulic.suspended_gold_g = self
            .mobile
            .iter()
            .zip(self.local_precision)
            .map(|(a, b)| a + b)
            .collect();
        Ok(())
    }
}
fn exchange(bed: &mut f64, mobile: &mut f64, before: f64, delta: f64, remaining: f64) {
    if delta > 0. {
        let fraction = (delta / (delta + remaining).max(1e-300)).clamp(0., 1.);
        let moved = *mobile * fraction;
        *mobile -= moved;
        *bed += moved;
    } else if delta < 0. && before > 0. {
        let moved = *bed * (-delta / before).clamp(0., 1.);
        *bed -= moved;
        *mobile += moved;
    }
}
fn phi(r: f64) -> f64 {
    if r < 0.01 {
        1. + r * (-0.5 + r * (1. / 6. + r * (-1. / 24. + r / 120.)))
    } else {
        -(-r).exp_m1() / r
    }
}
#[allow(clippy::too_many_arguments)]
fn reactive_advection(
    mobile: &mut [f64],
    bed: &mut [f64],
    exported: &mut f64,
    records: &[[f32; 4]],
    next_h: &[f64],
    width: usize,
    area: f64,
    dt: f64,
    ws: f64,
) -> Result<(), String> {
    let old = mobile.to_vec();
    mobile.fill(0.);
    let reaction = |i: usize| ws * dt / (0.5 * (records[i][0] as f64 + next_h[i])).max(1e-30);
    for (i, donor) in old.into_iter().enumerate() {
        if donor == 0. {
            continue;
        }
        let x = i % width;
        let z = i / width;
        let depth = records[i][0] as f64;
        let routes = [
            (
                Some(i + 1),
                if x + 1 < width {
                    (records[i][1] as f64).max(0.)
                } else {
                    0.
                },
            ),
            (
                i.checked_sub(1),
                if x > 0 {
                    -(records[i - 1][1] as f64).min(0.)
                } else {
                    0.
                },
            ),
            (
                Some(i + width),
                if z + 1 < width {
                    (records[i][2] as f64).max(0.)
                } else {
                    0.
                },
            ),
            (
                i.checked_sub(width),
                if z > 0 {
                    -(records[i - width][2] as f64).min(0.)
                } else {
                    0.
                },
            ),
            (None, records[i][3] as f64),
        ];
        let total = routes.iter().map(|(_, q)| *q).sum::<f64>();
        if depth < 0. || records[i][3] < 0. || (depth == 0. && total > 0.) {
            return Err("Invalid terrain water provenance event".into());
        }
        let fraction = if total > 0. {
            (total / (depth * area)).clamp(0., 1.)
        } else {
            0.
        };
        let r = reaction(i);
        let p = phi(r);
        let retained = donor * (1. - fraction) * (-r).exp();
        let departed = if r == 0. {
            donor - retained
        } else {
            donor * fraction * p
        };
        mobile[i] += retained;
        bed[i] += (donor - retained - departed).max(0.);
        let count = routes.iter().filter(|(_, q)| *q > 0.).count();
        let mut used = 0;
        let mut left = departed;
        for (target, q) in routes {
            if q <= 0. {
                continue;
            }
            used += 1;
            let moved = if used == count {
                left
            } else {
                (departed * (q / total)).min(left)
            };
            left -= moved;
            if let Some(j) = target {
                let other = reaction(j);
                let survival = (-r.min(other)).exp() * phi((r - other).abs());
                let arriving = moved * (survival / p).clamp(0., 1.);
                mobile[j] += arriving;
                bed[j] += moved - arriving;
            } else {
                *exported += moved;
            }
        }
    }
    Ok(())
}
#[cfg(test)]
fn advect(
    mobile: &mut [f64],
    exported: &mut f64,
    records: &[[f32; 4]],
    width: usize,
    area: f64,
) -> Result<(), String> {
    let mut bed = vec![0.; mobile.len()];
    let next_h = vec![1.; mobile.len()];
    reactive_advection(
        mobile, &mut bed, exported, records, &next_h, width, area, 0.05, 0.,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn substep_mixing_routes_new_arrivals_on_the_next_substep() {
        let mut gold = vec![0.; 9];
        gold[0] = 1.;
        let mut first = vec![[1., 0., 0., 0.]; 9];
        first[0][1] = 0.5;
        let mut second = vec![[1., 0., 0., 0.]; 9];
        second[0][0] = 0.5;
        second[1][0] = 1.5;
        second[1][1] = 0.75;
        let mut exported = 0.;
        advect(&mut gold, &mut exported, &first, 3, 1.).unwrap();
        advect(&mut gold, &mut exported, &second, 3, 1.).unwrap();
        assert_eq!(&gold[..3], &[0.5, 0.25, 0.25]);
        assert_eq!(gold.iter().sum::<f64>(), 1.);
        second[2][3] = 1.;
        advect(&mut gold, &mut exported, &second, 3, 1.).unwrap();
        assert_eq!(exported, 0.25);
        assert_eq!(gold.iter().sum::<f64>() + exported, 1.);
    }
    #[test]
    fn settle_then_scour_mixes_provenance_even_when_net_bed_height_is_unchanged() {
        let (mut bed, mut mobile) = (10., 2.);
        exchange(&mut bed, &mut mobile, 10., 1., 0.);
        exchange(&mut bed, &mut mobile, 11., -1., 1.);
        assert!((bed - 120. / 11.).abs() < 1e-14);
        assert!((mobile - 12. / 11.).abs() < 1e-14);
        assert_eq!(bed + mobile, 12.);
    }
    #[test]
    fn uniform_reaction_preserves_interior_mobile_and_bed_decay() {
        let mut mobile = vec![1.; 9];
        let mut bed = vec![0.; 9];
        let mut exported = 0.;
        let mut event = vec![[1., 0.2, 0., 0.]; 9];
        for i in [2, 5, 8] {
            event[i][1] = 0.;
            event[i][3] = 0.2;
        }
        reactive_advection(
            &mut mobile,
            &mut bed,
            &mut exported,
            &event,
            &[1.; 9],
            3,
            1.,
            0.5,
            2.,
        )
        .unwrap();
        assert!((mobile[1] - (-1_f64).exp()).abs() < 1e-15);
        assert!((bed[1] - (1. - (-1_f64).exp())).abs() < 1e-15);
        assert!(
            (mobile.iter().sum::<f64>() + bed.iter().sum::<f64>() + exported - 9.).abs() < 1e-14
        );
    }
    #[test]
    fn extremely_fast_reaction_remains_positive_conservative_without_microsteps() {
        let mut mobile = vec![1.];
        let mut bed = vec![0.];
        let mut exported = 0.;
        reactive_advection(
            &mut mobile,
            &mut bed,
            &mut exported,
            &[[1., 0., 0., 0.5]],
            &[1.],
            1,
            1.,
            1.,
            1e12,
        )
        .unwrap();
        assert_eq!(mobile[0], 0.);
        assert!(bed[0] > 0.99999999999);
        assert!(exported > 0.);
        assert!((bed[0] + exported - 1.).abs() < 1e-15);
    }
    #[test]
    fn cross_depth_arrival_matches_independent_time_quadrature() {
        let mut mobile = vec![1., 0., 0., 0.];
        let mut bed = vec![0.; 4];
        let mut exported = 0.;
        let q = 0.2_f32;
        let records = [
            [1., q, 0., 0.],
            [2., 0., 0., 0.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
        ];
        let next = [1. - q as f64, 2. + q as f64, 1., 1.];
        reactive_advection(
            &mut mobile,
            &mut bed,
            &mut exported,
            &records,
            &next,
            2,
            1.,
            1.,
            1.,
        )
        .unwrap();
        let a = 1. / ((1. + next[0]) * 0.5);
        let b = 1. / ((2. + next[1]) * 0.5);
        let mut integrated = 0.;
        let mut left = 0.;
        for i in 0..=1024 {
            let t = i as f64 / 1024.;
            let w = if i == 0 || i == 1024 {
                1.
            } else if i % 2 == 0 {
                2.
            } else {
                4.
            };
            integrated += w * (-a * t - b * (1. - t)).exp();
            left += w * (-a * t).exp();
        }
        let arriving = q as f64 * integrated / (3. * 1024.);
        let departed = q as f64 * left / (3. * 1024.);
        assert!((mobile[1] - arriving).abs() < 1e-14);
        assert!((bed[1] - (departed - arriving)).abs() < 1e-14);
    }
}
