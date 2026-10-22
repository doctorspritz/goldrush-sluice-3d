//! Continuous, shore-clipped presentation of the saved terrain-flow surface.
//! Additional vertices and flow-mapped optical detail do not add water volume.
use crate::water::InstanceData;
use mining_sim::{
    site::{CELL, WIDTH},
    Site,
};
pub(crate) const MAX_SPLATS: usize = (WIDTH - 1) * (WIDTH - 1);
// Presentation only; keep in step with MIN_FILM_DEPTH in water_depth.wgsl.
const MIN_VISIBLE_DEPTH_M: f32 = 0.0001;
pub(crate) fn instances(site: &Site, detail: &crate::environment::Detail) -> Vec<InstanceData> {
    let surface = &site.world.water_surface;
    let mut height = surface.clone();
    let mut supported = vec![false; WIDTH * WIDTH];
    let mut concentration = vec![0.; WIDTH * WIDTH];
    for z in 0..WIDTH {
        for x in 0..WIDTH {
            let i = z * WIDTH + x;
            supported[i] = site.world.water_depth(x, z) > MIN_VISIBLE_DEPTH_M;
            // Terrain transport stores reference bulk/water concentration.
            // Only actual mineral solids contribute to optical extinction.
            concentration[i] = [
                &site.hydraulic.suspended_sediment,
                &site.hydraulic.suspended_overburden,
                &site.hydraulic.suspended_gravel,
                &site.hydraulic.suspended_paydirt,
            ]
            .iter()
            .zip(mining_sim::material::REFERENCE_DRY_DENSITIES_KG_M3)
            .map(|(a, rho)| {
                (a.get(i).copied().unwrap_or(0.) as f64 * rho
                    / mining_sim::material::MINERAL_DENSITY_KG_M3) as f32
            })
            .sum::<f32>()
            .clamp(0., 1.);
            // Extend wet levels across the dry shore for reconstruction. The final
            // surface is clipped against actual ground, never draped up dry banks.
            if !supported[i] {
                let mut sum = 0.;
                let mut weight = 0.;
                for dz in -1..=1 {
                    for dx in -1..=1 {
                        let nx = x as i32 + dx;
                        let nz = z as i32 + dz;
                        if nx < 0 || nz < 0 || nx >= WIDTH as i32 || nz >= WIDTH as i32 {
                            continue;
                        }
                        if site.world.water_depth(nx as usize, nz as usize) > MIN_VISIBLE_DEPTH_M {
                            let w = 1. / (dx * dx + dz * dz).max(1) as f32;
                            sum += surface[nz as usize * WIDTH + nx as usize] * w;
                            weight += w;
                        }
                    }
                }
                if weight > 0. {
                    height[i] = sum / weight;
                    supported[i] = true;
                }
            }
        }
    }
    let mut slopes_x = vec![0.; WIDTH * WIDTH];
    let mut slopes_z = vec![0.; WIDTH * WIDTH];
    for z in 0..WIDTH {
        for x in 0..WIDTH {
            let lo = x.saturating_sub(1);
            let hi = (x + 1).min(WIDTH - 1);
            slopes_x[z * WIDTH + x] = (if supported[z * WIDTH + hi] {
                height[z * WIDTH + hi]
            } else {
                height[z * WIDTH + x]
            } - if supported[z * WIDTH + lo] {
                height[z * WIDTH + lo]
            } else {
                height[z * WIDTH + x]
            }) / ((hi - lo) as f32 * CELL);
            let lo = z.saturating_sub(1);
            let hi = (z + 1).min(WIDTH - 1);
            slopes_z[z * WIDTH + x] = (if supported[hi * WIDTH + x] {
                height[hi * WIDTH + x]
            } else {
                height[z * WIDTH + x]
            } - if supported[lo * WIDTH + x] {
                height[lo * WIDTH + x]
            } else {
                height[z * WIDTH + x]
            }) / ((hi - lo) as f32 * CELL);
        }
    }
    let mut result = Vec::new();
    let mut coarse = Vec::new();
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
                .any(|&i| site.world.water_depth(i % WIDTH, i / WIDTH) > MIN_VISIBLE_DEPTH_M)
            {
                continue;
            }
            let h = ids.map(|i| height[i]);
            // Extending the surface plane across shore must not turn a saved
            // dry corner into a visible film, especially below a millimetre.
            let depth = ids.map(|i| {
                site.world
                    .water_depth(i % WIDTH, i / WIDTH)
                    .min(height[i] - site.world.ground_height(i % WIDTH, i / WIDTH))
            });
            let vx = ids.map(|i| site.hydraulic.velocity_x.get(i).copied().unwrap_or(0.));
            let vz = ids.map(|i| site.hydraulic.velocity_z.get(i).copied().unwrap_or(0.));
            let subdivisions = detail.subdivisions(x, z);
            let instance = InstanceData {
                center_radius: [
                    (x as f32 + 0.5) * CELL,
                    0.,
                    (z as f32 + 0.5) * CELL,
                    CELL * 0.5,
                ],
                velocity: vx,
                shape: [0., 0., 1., subdivisions as f32],
                axis_x: h,
                axis_y: depth,
                axis_z: ids.map(|i| concentration[i]),
                slope_x: ids.map(|i| slopes_x[i]),
                slope_z: ids.map(|i| slopes_z[i]),
                flow_z: vz,
                ..Default::default()
            };
            if subdivisions > 1 {
                result.push(instance);
            } else {
                coarse.push(instance);
            }
        }
    }
    result.extend(coarse);
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn near_detail_keeps_instances_bounded_and_updates_on_paused_camera_moves() {
        let site = Site::new();
        let mut detail = crate::environment::Detail::default();
        detail.select(bevy::prelude::Vec3::new(7., 2., 17.));
        let a = instances(&site, &detail);
        let fine = a.iter().take_while(|p| p.shape[3] > 1.).count();
        assert!(
            fine > 0
                && fine <= crate::environment::MAX_FINE_CHUNKS * crate::environment::CHUNK.pow(2)
        );
        assert!(a[fine..].iter().all(|p| p.shape[3] == 1.));
        assert!(a.len() <= MAX_SPLATS);
        let time = site.hydraulic.steps;
        let revision = detail.revision;
        detail.select(bevy::prelude::Vec3::new(28., 2., 17.));
        let b = instances(&site, &detail);
        assert_eq!(a.len(), b.len());
        assert_eq!(time, site.hydraulic.steps);
        assert_ne!(revision, detail.revision);
        assert!(b.iter().filter(|p| p.shape[3] > 1.).count() < fine);
    }
    #[test]
    fn shoreline_extension_keeps_saved_dry_corners_dry() {
        let mut site = Site::new();
        for z in 0..WIDTH {
            for x in 0..WIDTH {
                site.world.water_surface[z * WIDTH + x] = site.world.ground_height(x, z);
            }
        }
        site.world.water_surface[60 * WIDTH + 80] += 0.003;
        let water = instances(&site, &crate::environment::Detail::default());
        assert_eq!(water.len(), 4);
        assert!(water
            .iter()
            .all(|p| p.axis_y.iter().filter(|&&d| d > 0.).count() == 1));
    }
    #[test]
    fn submillimetre_films_have_bounded_coverage_without_adding_water() {
        let mut site = Site::new();
        for z in 0..WIDTH {
            for x in 0..WIDTH {
                site.world.water_surface[z * WIDTH + x] = site.world.ground_height(x, z) + 0.0005;
            }
        }
        let before = site.world.water_surface.clone();
        let film = instances(&site, &crate::environment::Detail::default());
        assert_eq!(film.len(), MAX_SPLATS);
        assert_eq!(before, site.world.water_surface);
        assert!(film
            .iter()
            .all(|p| p.axis_y.iter().all(|h| (*h - 0.0005).abs() < 0.000001)));
        // Numerical residue must not turn the entire dry site into water.
        for z in 0..WIDTH {
            for x in 0..WIDTH {
                site.world.water_surface[z * WIDTH + x] = site.world.ground_height(x, z) + 0.00005;
            }
        }
        assert!(instances(&site, &crate::environment::Detail::default()).is_empty());
    }
    #[test]
    fn continuous_cells_share_vertices_and_do_not_mutate_water() {
        let mut site = Site::new();
        site.hydraulic.velocity_x = (0..WIDTH * WIDTH)
            .map(|i| (i % WIDTH) as f32 * 0.001)
            .collect();
        site.hydraulic.velocity_z = (0..WIDTH * WIDTH)
            .map(|i| (i / WIDTH) as f32 * -0.001)
            .collect();
        let before = site.world.water_surface.clone();
        let a = instances(&site, &crate::environment::Detail::default());
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
                assert_eq!(p.slope_x[1], next.slope_x[0]);
                assert_eq!(p.slope_z[3], next.slope_z[2]);
                assert_eq!(p.velocity[1], next.velocity[0]);
                assert_eq!(p.flow_z[3], next.flow_z[2]);
            }
        }
        let b = instances(&site, &crate::environment::Detail::default());
        assert!(a
            .iter()
            .zip(b)
            .all(|(a, b)| a.center_radius == b.center_radius && a.axis_x == b.axis_x));
    }
    #[test]
    fn concentration_is_independent_of_water_depth_and_dry_cells_are_absent() {
        let mut site = Site::new();
        site.hydraulic.suspended_sediment = vec![0.02; WIDTH * WIDTH];
        let a = instances(&site, &crate::environment::Detail::default());
        let expected = (0.02_f32 as f64 * 1600. / 2650.) as f32;
        assert!(a.iter().all(|p| p.axis_z == [expected; 4]));
        for z in 0..WIDTH {
            for x in 0..WIDTH {
                let i = z * WIDTH + x;
                site.world.water_surface[i] = site.world.ground_height(x, z);
            }
        }
        assert!(instances(&site, &crate::environment::Detail::default()).is_empty());
    }
}
