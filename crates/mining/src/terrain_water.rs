//! Continuous, shore-clipped presentation of the saved terrain-flow surface.
//! Additional vertices and flow-mapped optical detail do not add water volume.
use crate::water::InstanceData;
use mining_sim::{
    site::{CELL, WIDTH},
    Site,
};
pub(crate) const MAX_SPLATS: usize = (WIDTH - 1) * (WIDTH - 1);
pub(crate) fn instances(site: &Site, _sim: f64) -> Vec<InstanceData> {
    let surface = &site.world.water_surface;
    let mut height = surface.clone();
    let mut concentration = vec![0.; WIDTH * WIDTH];
    for z in 0..WIDTH {
        for x in 0..WIDTH {
            let i = z * WIDTH + x;
            // These channels already contain volume concentration, not sediment depth.
            concentration[i] = [
                &site.hydraulic.suspended_sediment,
                &site.hydraulic.suspended_overburden,
                &site.hydraulic.suspended_gravel,
                &site.hydraulic.suspended_paydirt,
            ]
            .iter()
            .map(|a| a.get(i).copied().unwrap_or(0.))
            .sum::<f32>()
            .clamp(0., 1.);
            // Extend wet levels across the dry shore for reconstruction. The final
            // surface is clipped against actual ground, never draped up dry banks.
            if site.world.water_depth(x, z) < 0.001 {
                let mut sum = 0.;
                let mut weight = 0.;
                for dz in -1..=1 {
                    for dx in -1..=1 {
                        let nx = x as i32 + dx;
                        let nz = z as i32 + dz;
                        if nx < 0 || nz < 0 || nx >= WIDTH as i32 || nz >= WIDTH as i32 {
                            continue;
                        }
                        if site.world.water_depth(nx as usize, nz as usize) > 0.001 {
                            let w = 1. / (dx * dx + dz * dz).max(1) as f32;
                            sum += surface[nz as usize * WIDTH + nx as usize] * w;
                            weight += w;
                        }
                    }
                }
                if weight > 0. {
                    height[i] = sum / weight;
                }
            }
        }
    }
    let mut result = Vec::new();
    for z in 0..WIDTH - 1 {
        for x in 0..WIDTH - 1 {
            let ids = [
                z * WIDTH + x,
                z * WIDTH + x + 1,
                (z + 1) * WIDTH + x,
                (z + 1) * WIDTH + x + 1,
            ];
            if !ids
                .iter()
                .any(|&i| site.world.water_depth(i % WIDTH, i / WIDTH) > 0.001)
            {
                continue;
            }
            let h = ids.map(|i| height[i]);
            let depth = ids.map(|i| height[i] - site.world.ground_height(i % WIDTH, i / WIDTH));
            let vx = ids
                .iter()
                .map(|i| site.hydraulic.velocity_x.get(*i).copied().unwrap_or(0.))
                .sum::<f32>()
                * 0.25;
            let vz = ids
                .iter()
                .map(|i| site.hydraulic.velocity_z.get(*i).copied().unwrap_or(0.))
                .sum::<f32>()
                * 0.25;
            result.push(InstanceData {
                center_radius: [
                    (x as f32 + 0.5) * CELL,
                    0.,
                    (z as f32 + 0.5) * CELL,
                    CELL * 0.5,
                ],
                velocity: [vx, 0., vz, 0.],
                shape: [0., 0., 1., 0.],
                axis_x: h,
                axis_y: depth,
                axis_z: ids.map(|i| concentration[i]),
            });
        }
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn continuous_cells_share_vertices_and_do_not_mutate_water() {
        let site = Site::new();
        let before = site.world.water_surface.clone();
        let a = instances(&site, 0.);
        assert!(!a.is_empty() && a.len() <= MAX_SPLATS);
        assert_eq!(before, site.world.water_surface);
        let lookup: std::collections::BTreeMap<_, _> = a
            .iter()
            .map(|p| {
                (
                    (
                        (p.center_radius[0] / CELL) as usize,
                        (p.center_radius[2] / CELL) as usize,
                    ),
                    p,
                )
            })
            .collect();
        for (&(x, z), p) in &lookup {
            if let Some(next) = lookup.get(&(x + 1, z)) {
                assert_eq!(p.axis_x[1], next.axis_x[0]);
                assert_eq!(p.axis_x[3], next.axis_x[2]);
            }
        }
        let b = instances(&site, 1000.);
        assert!(a
            .iter()
            .zip(b)
            .all(|(a, b)| a.center_radius == b.center_radius && a.axis_x == b.axis_x));
    }
    #[test]
    fn concentration_is_independent_of_water_depth_and_dry_cells_are_absent() {
        let mut site = Site::new();
        site.hydraulic.suspended_sediment = vec![0.02; WIDTH * WIDTH];
        let a = instances(&site, 0.);
        assert!(a.iter().all(|p| p.axis_z == [0.02; 4]));
        for z in 0..WIDTH {
            for x in 0..WIDTH {
                let i = z * WIDTH + x;
                site.world.water_surface[i] = site.world.ground_height(x, z);
            }
        }
        assert!(instances(&site, 0.).is_empty());
    }
}
