//! Conservative channel-to-terrain handoff. FLIP/APIC and DEM own the channel;
//! outside its lip water follows gravity until its first ground/water impact.
use crate::{
    site::{CELL, WIDTH},
    Site, Sluice, Vec3,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutflowParticle {
    #[serde(default)]
    pub id: u64,
    pub position: Vec3,
    pub velocity: Vec3,
}

/// Preserve the actual exit state; never rotate gravity by the channel slope,
/// prescribe a discharge speed, or return airborne water at a fixed location.
pub fn release(s: &mut Sluice) {
    let Some(placement) = s.placement else {
        return;
    };
    let length = s.profile.length();
    let mut water = std::mem::take(&mut s.water);
    water.retain(|p| {
        if p.position.x < length {
            return true;
        }
        s.outflow.push(OutflowParticle {
            id: p.id,
            position: placement.to_world(s.grid_to_local(p.position)),
            velocity: s.grid_velocity_to_world(p.velocity),
        });
        false
    });
    s.water = water;
}
fn trajectory(p: &OutflowParticle, t: f32) -> Vec3 {
    p.position + p.velocity * t - Vec3::Y * (0.5 * 9.81 * t * t)
}
fn surface(site: &Site, p: Vec3) -> f32 {
    // The same bilinear convention as Site::height and the terrain renderer.
    let x = (p.x / CELL).clamp(0., (WIDTH - 1) as f32);
    let z = (p.z / CELL).clamp(0., (WIDTH - 1) as f32);
    let ix = x as usize;
    let iz = z as usize;
    let nx = (ix + 1).min(WIDTH - 1);
    let nz = (iz + 1).min(WIDTH - 1);
    let w = &site.world.water_surface;
    let a = w[iz * WIDTH + ix] * (1. - x.fract()) + w[iz * WIDTH + nx] * x.fract();
    let b = w[nz * WIDTH + ix] * (1. - x.fract()) + w[nz * WIDTH + nx] * x.fract();
    (a * (1. - z.fract()) + b * z.fract()).max(site.height(p.x, p.z))
}

pub fn step(s: &mut Sluice, site: &mut Site, dt: f32) {
    if s.outflow.is_empty() || dt <= 0. {
        return;
    }
    // Aggregate tiny particle volumes before converting to f32 terrain depths.
    let volume = s.profile.particle_m3();
    let mut deposits = BTreeMap::<usize, (f64, f64, f64)>::new();
    let mut landed = 0;
    s.outflow.retain_mut(|p| {
        let travel = p.velocity.x.abs().max(p.velocity.z.abs()) * dt;
        let samples = (travel / (CELL * 0.25)).ceil().max(1.) as usize;
        let mut previous = 0.;
        let mut hit = None;
        for n in 0..=samples {
            let t = dt * n as f32 / samples as f32;
            let at = trajectory(p, t);
            if at.y <= surface(site, at) {
                let mut lo = previous;
                let mut hi = t;
                for _ in 0..12 {
                    let middle = (lo + hi) * 0.5;
                    let q = trajectory(p, middle);
                    if q.y <= surface(site, q) {
                        hi = middle;
                    } else {
                        lo = middle;
                    }
                }
                hit = Some(hi);
                break;
            }
            previous = t;
        }
        if let Some(t) = hit {
            let at = trajectory(p, t);
            let x = (at.x / CELL).clamp(0., (WIDTH - 1) as f32);
            let z = (at.z / CELL).clamp(0., (WIDTH - 1) as f32);
            let ix = x as usize;
            let iz = z as usize;
            for (cx, cz, weight) in [
                (ix, iz, (1. - x.fract()) * (1. - z.fract())),
                ((ix + 1).min(WIDTH - 1), iz, x.fract() * (1. - z.fract())),
                (ix, (iz + 1).min(WIDTH - 1), (1. - x.fract()) * z.fract()),
                (
                    (ix + 1).min(WIDTH - 1),
                    (iz + 1).min(WIDTH - 1),
                    x.fract() * z.fract(),
                ),
            ] {
                let v = volume * weight as f64;
                let entry = deposits.entry(cz * WIDTH + cx).or_default();
                entry.0 += v;
                entry.1 += v * p.velocity.x as f64;
                entry.2 += v * p.velocity.z as f64;
            }
            landed += 1;
            false
        } else {
            p.position = trajectory(p, dt);
            p.velocity.y -= 9.81 * dt;
            true
        }
    });
    if landed == 0 {
        return;
    }
    let a = &mut site.hydraulic;
    if a.velocity_x.is_empty() {
        a.velocity_x = vec![0.; WIDTH * WIDTH];
    }
    if a.velocity_z.is_empty() {
        a.velocity_z = vec![0.; WIDTH * WIDTH];
    }
    for (i, (v, mx, mz)) in deposits {
        let old = site.world.water_depth(i % WIDTH, i / WIDTH) as f64 * (CELL * CELL) as f64;
        a.velocity_x[i] = ((old * a.velocity_x[i] as f64 + mx) / (old + v).max(1e-20)) as f32;
        a.velocity_z[i] = ((old * a.velocity_z[i] as f64 + mz) / (old + v).max(1e-20)) as f32;
        let dilution = (old / (old + v).max(1e-20)) as f32;
        for channel in [
            &mut a.suspended_sediment,
            &mut a.suspended_overburden,
            &mut a.suspended_gravel,
            &mut a.suspended_paydirt,
        ] {
            if let Some(concentration) = channel.get_mut(i) {
                *concentration *= dilution;
            }
        }
        site.world.water_surface[i] += (v / (CELL * CELL) as f64) as f32;
    }
    a.impact_dirty = true;
    s.water_returned_m3 += landed as f64 * volume;
}
