//! Conservative f64 passive provenance for coarse paydirt. Replay the actual
//! water and bed events; a sum of face volumes cannot reproduce substep mixing.
//! Gold separation in the sluice remains the FLIP/DEM model.
use mining_sim::{site::WIDTH, Site};

pub(super) struct TerrainGold {
    mobile: Vec<f64>,
    local_precision: Vec<f64>,
    bed_high: Vec<f32>,
    bed_low: Vec<f64>,
}
impl TerrainGold {
    pub(super) fn new(site: &Site, depth: &[f32], area: f64) -> Self {
        let mut mobile = vec![0.; depth.len()];
        let mut local_precision = vec![0.; depth.len()];
        let bed_high = site.world.paydirt_thickness.clone();
        let mut bed_low = vec![0.; depth.len()];
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
            bed_low[i] = site
                .hydraulic
                .bed_remainder_m3
                .get(i)
                .map(|v| v[3] / area)
                .unwrap_or(0.);
        }
        Self {
            mobile,
            local_precision,
            bed_high,
            bed_low,
        }
    }
    pub(super) fn replay(
        mut self,
        site: &mut Site,
        events: &[[f32; 4]],
        substeps: usize,
        retained: &[[f64; 4]],
        area: f64,
    ) -> Result<(), String> {
        let n = WIDTH * WIDTH;
        if events.len() != (substeps + 2) * n || events.iter().flatten().any(|v| !v.is_finite()) {
            return Err("Invalid terrain provenance events".into());
        }
        for event in events[..substeps * n].chunks_exact(n) {
            advect(
                &mut self.mobile,
                &mut site.hydraulic.exported_gold_g,
                event,
                WIDTH,
                area,
            )?;
        }
        for event in events[substeps * n..].chunks_exact(n) {
            for (i, record) in event.iter().enumerate() {
                if record.iter().any(|v| *v < 0.) {
                    return Err("Negative terrain bed provenance event".into());
                }
                let next_low = record[1] as f64 + retained[i][3] / area;
                // Subtract the two parts separately: adding a large geometric
                // height first would erase a tiny but counted bed transfer.
                let delta =
                    (record[0] as f64 - self.bed_high[i] as f64) + (next_low - self.bed_low[i]);
                exchange(
                    &mut site.gold_g[i],
                    &mut self.mobile[i],
                    self.bed_high[i] as f64 + self.bed_low[i],
                    delta,
                    record[2] as f64 * record[3] as f64,
                );
                self.bed_high[i] = record[0];
                self.bed_low[i] = next_low;
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
fn advect(
    mobile: &mut [f64],
    exported: &mut f64,
    records: &[[f32; 4]],
    width: usize,
    area: f64,
) -> Result<(), String> {
    let old = mobile.to_vec();
    mobile.fill(0.);
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
        let budget = if total > 0. {
            donor * (total / (depth * area)).clamp(0., 1.)
        } else {
            0.
        };
        mobile[i] += donor - budget;
        let count = routes.iter().filter(|(_, q)| *q > 0.).count();
        let mut left = budget;
        let mut used = 0;
        for (target, q) in routes {
            if q <= 0. {
                continue;
            }
            used += 1;
            let moved = if used == count {
                left
            } else {
                (budget * (q / total)).min(left)
            };
            left -= moved;
            if let Some(j) = target {
                mobile[j] += moved;
            } else {
                *exported += moved;
            }
        }
    }
    Ok(())
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
}
