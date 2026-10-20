//! Render-only tangent-disc splats of the saved terrain-flow state. These are
//! neither new water volume nor a replacement for the conservative SWE solver.
use crate::water::InstanceData;
use mining_sim::{
    site::{CELL, WIDTH},
    Site,
};
pub(crate) const MAX_SPLATS: usize = WIDTH * WIDTH * 4;
pub(crate) fn instances(site: &Site, sim: f64) -> Vec<InstanceData> {
    let surface = &site.world.water_surface;
    let sample = |x: f32, z: f32| {
        let x = (x / CELL).clamp(0., (WIDTH - 1) as f32);
        let z = (z / CELL).clamp(0., (WIDTH - 1) as f32);
        let ix = x as usize;
        let iz = z as usize;
        let nx = (ix + 1).min(WIDTH - 1);
        let nz = (iz + 1).min(WIDTH - 1);
        let a = surface[iz * WIDTH + ix] * (1. - x.fract()) + surface[iz * WIDTH + nx] * x.fract();
        let b = surface[nz * WIDTH + ix] * (1. - x.fract()) + surface[nz * WIDTH + nx] * x.fract();
        a * (1. - z.fract()) + b * z.fract()
    };
    let mut result = Vec::new();
    for z in 0..WIDTH {
        for x in 0..WIDTH {
            let i = z * WIDTH + x;
            let depth = site.world.water_depth(x, z);
            if depth < 0.008 {
                continue;
            }
            let vx = site.hydraulic.velocity_x.get(i).copied().unwrap_or(0.);
            let vz = site.hydraulic.velocity_z.get(i).copied().unwrap_or(0.);
            let sediment = [
                &site.hydraulic.suspended_sediment,
                &site.hydraulic.suspended_overburden,
                &site.hydraulic.suspended_gravel,
                &site.hydraulic.suspended_paydirt,
            ]
            .iter()
            .map(|a| a.get(i).copied().unwrap_or(0.))
            .sum::<f32>();
            let concentration = (sediment / depth.max(0.001)).clamp(0., 1.);
            // Advect a periodic sampling pattern using physical simulation time and
            // saved velocity, rather than animating artificial surface-height waves.
            let phase_x = (sim * vx as f64 / CELL as f64).rem_euclid(1.) as f32;
            let phase_z = (sim * vz as f64 / CELL as f64).rem_euclid(1.) as f32;
            for qz in 0..2 {
                for qx in 0..2 {
                    let px = (x as f32 + (0.25 + qx as f32 * 0.5 + phase_x).fract()) * CELL;
                    let pz = (z as f32 + (0.25 + qz as f32 * 0.5 + phase_z).fract()) * CELL;
                    let height = sample(px, pz);
                    let local_depth = (height - site.height(px, pz)).max(0.);
                    if local_depth < 0.008 {
                        continue;
                    }
                    let slope_x = (sample(px + 0.01, pz) - sample(px - 0.01, pz)) / 0.02;
                    let slope_z = (sample(px, pz + 0.01) - sample(px, pz - 0.01)) / 0.02;
                    let vertical = vx * slope_x + vz * slope_z;
                    result.push(InstanceData {
                        center_radius: [px, height, pz, CELL * 0.56],
                        velocity: [vx, vertical, vz, 0.],
                        shape: [slope_x, concentration, 1., slope_z],
                    });
                }
            }
        }
    }
    result
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn terrain_splats_follow_water_and_do_not_mutate_its_volume() {
        let site = Site::new();
        let before = site.world.water_surface.clone();
        let a = instances(&site, 0.);
        assert!(!a.is_empty() && a.len() <= MAX_SPLATS);
        assert!(a.iter().all(|p| p.shape[2] == 1.
            && p.shape.iter().all(|n| n.is_finite())
            && p.center_radius[3] == CELL * 0.56
            && p.center_radius.iter().all(|n| n.is_finite())));
        assert_eq!(before, site.world.water_surface);
        let mut dry = site.clone();
        for z in 0..WIDTH {
            for x in 0..WIDTH {
                let i = z * WIDTH + x;
                dry.world.water_surface[i] = dry.world.ground_height(x, z);
            }
        }
        assert!(instances(&dry, 0.).is_empty());
    }
}
