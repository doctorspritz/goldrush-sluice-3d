//! Independent world gold after discharge. This is a dilute, one-way,
//! physical-diameter point-parcel approximation, not a calibrated bedload law.
//! SWE supplies depth-averaged horizontal flow; turbulence, shape drag, grain
//! collisions and reaction forces on water remain outside this model.
use crate::{
    site::{CELL, EXTENT, WIDTH},
    sluice::Grain,
    GrainKind, Material, Site,
};
use glam::{DVec3, Quat};
use serde::{Deserialize, Serialize};

pub const MAX_PARTICLES: usize = 65_536;
const MAX_SUBSTEPS: usize = 4096;
const MAX_WORK: usize = 2_000_000;
const G: f64 = 9.81;
const GOLD_DENSITY: f64 = 19_300.;
const WATER_DENSITY: f64 = 1000.;
const CONTACT_EPS: f64 = 2e-7;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoldParticle {
    pub id: u64,
    pub grain: Grain,
    /// Authoritative world motion. Grain's f32 fields are exact display mirrors.
    pub position: DVec3,
    pub velocity: DVec3,
    pub supported: bool,
    pub buried: bool,
    /// Only unsupported submerged grains participate in mixed pump withdrawal.
    pub mobile: bool,
    pub last_bed_height: Option<f64>,
    /// Admission time within the pending fixed step; saved and consumed once.
    pub delay_s: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LooseGold {
    pub model_version: u32,
    /// Explicit prototype Coulomb coefficient, not a measured site material.
    pub friction_mu: f64,
    pub particles: Vec<GoldParticle>,
    pub next_id: u64,
    pub exported_g: f64,
    pub admitted_g: f64,
    pub excavated_g: f64,
    pub withdrawn_g: f64,
}
impl Default for LooseGold {
    fn default() -> Self {
        Self {
            model_version: 1,
            friction_mu: 0.6,
            particles: Vec::new(),
            next_id: 0,
            exported_g: 0.,
            admitted_g: 0.,
            excavated_g: 0.,
            withdrawn_g: 0.,
        }
    }
}

fn pure_grain(g: &Grain) -> bool {
    g.kind == GrainKind::Gold
        && g.material.valid()
        && g.material.gold_g > 0.
        && g.material.volume() == 0.
        && g.material.pore_water_m3 == 0.
        && (0.00002..=0.02).contains(&g.diameter_m)
        && g.position.is_finite()
        && g.velocity.is_finite()
        && g.affine.is_finite()
        && g.rotation.is_finite()
        && (g.rotation.length_squared() - 1.).abs() < 1e-3
        && g.angular_velocity.is_finite()
}
impl LooseGold {
    pub fn try_reserve(&mut self, count: usize) -> Result<(), String> {
        if self
            .particles
            .len()
            .checked_add(count)
            .is_none_or(|n| n > MAX_PARTICLES)
            || self.next_id.checked_add(count as u64).is_none()
        {
            return Err("Independent gold capacity exceeded; donor retained".into());
        }
        self.particles
            .try_reserve(count)
            .map_err(|_| "Cannot allocate independent gold; donor retained".into())
    }
    pub fn admit(&mut self, grain: &Grain) -> Result<(), String> {
        self.admit_at(grain, 0.)
    }
    pub fn admit_at(&mut self, grain: &Grain, delay_s: f64) -> Result<(), String> {
        if !pure_grain(grain)
            || !delay_s.is_finite()
            || !(0. ..=0.25).contains(&delay_s)
            || !valid_position(grain.position.as_dvec3())
            || !(self.admitted_g + grain.material.gold_g).is_finite()
        {
            return Err("Independent gold requires a finite pure-gold world grain".into());
        }
        self.try_reserve(1)?;
        self.next_id += 1;
        self.admitted_g += grain.material.gold_g;
        self.particles.push(GoldParticle {
            id: self.next_id,
            grain: grain.clone(),
            position: grain.position.as_dvec3(),
            velocity: grain.velocity.as_dvec3(),
            supported: grain.resting,
            buried: false,
            mobile: false,
            last_bed_height: None,
            delay_s,
        });
        Ok(())
    }
    pub fn material(&self) -> Material {
        Material {
            gold_g: self.particles.iter().map(|p| p.grain.material.gold_g).sum(),
            ..Material::default()
        }
    }
    /// Accept and advance just this new packet through the remainder of the
    /// current fixed step. Existing particles have already consumed their dt.
    /// All fallible physics runs on a local particle before ownership changes.
    pub fn admit_and_advance(&mut self, grain: &Grain, site: &Site, dt: f32) -> Result<(), String> {
        if !dt.is_finite()
            || !(0. ..=0.25).contains(&dt)
            || !self.valid()
            || !pure_grain(grain)
            || !valid_position(grain.position.as_dvec3())
        {
            return Err("Invalid independent gold admission or remainder timestep".into());
        }
        let id = self
            .next_id
            .checked_add(1)
            .ok_or("Independent gold identity capacity exceeded")?;
        if self.particles.len() >= MAX_PARTICLES {
            return Err("Independent gold capacity exceeded; donor retained".into());
        }
        let admitted = self.admitted_g + grain.material.gold_g;
        if !admitted.is_finite() {
            return Err("Independent gold receipt overflows; donor retained".into());
        }
        let mut particle = GoldParticle {
            id,
            grain: grain.clone(),
            position: grain.position.as_dvec3(),
            velocity: grain.velocity.as_dvec3(),
            supported: grain.resting,
            buried: false,
            mobile: false,
            last_bed_height: None,
            delay_s: 0.,
        };
        let retained =
            dt == 0. || advance(&mut particle, site, dt as f64, self.friction_mu, &mut 0)?;
        if retained
            && (!pure_grain(&particle.grain)
                || !valid_position(particle.position)
                || !particle.velocity.is_finite()
                || particle.grain.position != particle.position.as_vec3()
                || particle.grain.velocity != particle.velocity.as_vec3())
        {
            return Err("Invalid independent gold continuation; donor retained".into());
        }
        if retained {
            self.try_reserve(1)?;
            self.particles.push(particle);
        } else {
            self.exported_g += grain.material.gold_g;
        }
        self.next_id = id;
        self.admitted_g = admitted;
        Ok(())
    }
    pub fn valid(&self) -> bool {
        let mut ids = std::collections::HashSet::new();
        let stocks = [
            self.exported_g,
            self.admitted_g,
            self.excavated_g,
            self.withdrawn_g,
        ];
        let closure =
            self.material().gold_g + self.exported_g + self.excavated_g + self.withdrawn_g;
        self.model_version == 1
            && self.friction_mu.is_finite()
            && (0. ..=2.).contains(&self.friction_mu)
            && self.particles.len() <= MAX_PARTICLES
            && stocks.iter().all(|v| v.is_finite() && *v >= 0.)
            && closure.is_finite()
            && (closure - self.admitted_g).abs() <= (self.admitted_g * 1e-9).max(1e-15)
            && self.particles.iter().all(|p| {
                p.id > 0
                    && p.id <= self.next_id
                    && ids.insert(p.id)
                    && pure_grain(&p.grain)
                    && valid_position(p.position)
                    && p.velocity.is_finite()
                    && p.position.as_vec3() == p.grain.position
                    && p.velocity.as_vec3() == p.grain.velocity
                    && p.last_bed_height.is_none_or(f64::is_finite)
                    && p.delay_s.is_finite()
                    && (0. ..=0.25).contains(&p.delay_s)
                    && p.grain.resting == (p.supported || p.buried)
                    && !(p.supported && p.buried)
                    && !(p.buried && p.mobile)
                    && !(p.supported && p.mobile)
            })
    }
    /// Transactional: numerical-work refusal leaves the entire owner unchanged.
    /// Return only this step's export; exported_g is the counted cumulative sink.
    pub fn step(&mut self, site: &Site, dt: f32) -> Result<f64, String> {
        if !dt.is_finite() || !(0. ..=0.25).contains(&dt) || !self.valid() {
            return Err("Invalid independent gold timestep or owner".into());
        }
        if dt == 0. {
            return Ok(0.);
        }
        let mut next = self.clone();
        let mut work = 0;
        let mut exported = 0.;
        for p in &mut next.particles {
            let time = (dt as f64 - p.delay_s).max(0.);
            p.delay_s = (p.delay_s - dt as f64).max(0.);
            if time == 0. {
                continue;
            }
            if !advance(p, site, time, self.friction_mu, &mut work)? {
                exported += p.grain.material.gold_g;
                // Mark for removal without replacing the parcel's identity.
                p.id = 0;
            }
        }
        next.particles.retain(|p| p.id != 0);
        next.exported_g += exported;
        if !next.valid() {
            return Err("Independent gold step produced invalid state; owner retained".into());
        }
        *self = next;
        Ok(exported)
    }
    /// Explicit aggregate excavation boundary. Call with the PRE-CUT terrain.
    /// Full grain identity is retained until this existing bulk-material handoff.
    pub fn preview_cut(&self, site: &Site, cuts: &[(usize, f32)]) -> f64 {
        let depths = cut_depths(site, cuts);
        self.particles
            .iter()
            .filter(|p| cut_contains(p, site, &depths))
            .map(|p| p.grain.material.gold_g)
            .sum()
    }
    pub fn take_cut(&mut self, site: &Site, cuts: &[(usize, f32)]) -> f64 {
        let depths = cut_depths(site, cuts);
        let mut taken = 0.;
        self.particles.retain(|p| {
            if cut_contains(p, site, &depths) {
                taken += p.grain.material.gold_g;
                false
            } else {
                true
            }
        });
        self.excavated_g += taken;
        taken
    }
    /// Well-mixed aggregate pump boundary, matching represented water removal.
    /// Bed-supported/buried gold is excluded; no mineral carrier is fabricated.
    pub fn withdraw_mobile(&mut self, cell: usize, fraction: f64) -> f64 {
        if cell >= WIDTH * WIDTH || !fraction.is_finite() || !(0. ..=1.).contains(&fraction) {
            return 0.;
        }
        let mut taken = 0.;
        for p in &mut self.particles {
            if p.mobile && !p.supported && !p.buried && particle_cell(p.position) == cell {
                let removed = p.grain.material.gold_g * fraction;
                p.grain.material.gold_g -= removed;
                taken += removed;
            }
        }
        self.particles.retain(|p| p.grain.material.gold_g > 0.);
        self.withdrawn_g += taken;
        taken
    }
}

fn valid_position(p: DVec3) -> bool {
    p.is_finite() && p.abs().max_element() < 1000.
}
fn in_claim(p: DVec3) -> bool {
    p.x >= 0. && p.z >= 0. && p.x < EXTENT as f64 && p.z < EXTENT as f64
}
fn particle_cell(p: DVec3) -> usize {
    let x = (p.x / CELL as f64).floor() as usize;
    let z = (p.z / CELL as f64).floor() as usize;
    z.min(WIDTH - 1) * WIDTH + x.min(WIDTH - 1)
}
/// Same diagonal as soil::substrate_height / Parry, evaluated in f64 so contact
/// does not repeatedly add f32 terrain rounding to a 20 micrometre grain.
pub(crate) fn bed_at(site: &Site, p: DVec3) -> (f64, DVec3) {
    let gx = (p.x / CELL as f64).clamp(0., (WIDTH - 1) as f64);
    let gz = (p.z / CELL as f64).clamp(0., (WIDTH - 1) as f64);
    let x = (gx.floor() as usize).min(WIDTH - 2);
    let z = (gz.floor() as usize).min(WIDTH - 2);
    let fx = gx - x as f64;
    let fz = gz - z as f64;
    let a = site.world.ground_height(x, z) as f64;
    let b = site.world.ground_height(x, z + 1) as f64;
    let c = site.world.ground_height(x + 1, z) as f64;
    let d = site.world.ground_height(x + 1, z + 1) as f64;
    let (height, dx, dz) = if fx + fz <= 1. {
        (a + (c - a) * fx + (b - a) * fz, c - a, b - a)
    } else {
        (d + (b - d) * (1. - fx) + (c - d) * (1. - fz), d - b, d - c)
    };
    (
        height,
        DVec3::new(-dx / CELL as f64, 1., -dz / CELL as f64).normalize(),
    )
}
fn water(site: &Site, p: DVec3) -> (f64, DVec3) {
    let gx = (p.x / CELL as f64).clamp(0., (WIDTH - 1) as f64);
    let gz = (p.z / CELL as f64).clamp(0., (WIDTH - 1) as f64);
    let x = gx.floor() as usize;
    let z = gz.floor() as usize;
    let nx = (x + 1).min(WIDTH - 1);
    let nz = (z + 1).min(WIDTH - 1);
    let fx = gx.fract();
    let fz = gz.fract();
    let mut surface = 0.;
    let mut wet_weight = 0.;
    let mut flow = DVec3::ZERO;
    for (x, z, weight) in [
        (x, z, (1. - fx) * (1. - fz)),
        (nx, z, fx * (1. - fz)),
        (x, nz, (1. - fx) * fz),
        (nx, nz, fx * fz),
    ] {
        let i = z * WIDTH + x;
        let depth = site.world.water_depth(x, z) as f64;
        surface += site.world.water_surface[i] as f64 * weight;
        if depth > 0. {
            let w = weight * depth;
            wet_weight += w;
            flow += DVec3::new(
                site.hydraulic.velocity_x.get(i).copied().unwrap_or(0.) as f64,
                0.,
                site.hydraulic.velocity_z.get(i).copied().unwrap_or(0.) as f64,
            ) * w;
        }
    }
    (
        if wet_weight > 0. {
            surface
        } else {
            f64::NEG_INFINITY
        },
        if wet_weight > 0. {
            flow / wet_weight
        } else {
            DVec3::ZERO
        },
    )
}
fn terminal_speed(diameter: f64) -> f64 {
    let relative = GOLD_DENSITY / WATER_DENSITY - 1.;
    relative * G * diameter.powi(2) / (18e-6 + (0.75 * relative * G * diameter.powi(3)).sqrt())
}
/// Exact velocity and displacement under constant sampled flow and linear drag.
fn motion(v: DVec3, flow: DVec3, dt: f64, diameter: f64, wet: bool) -> (DVec3, DVec3) {
    if !wet {
        let a = DVec3::new(0., -G, 0.);
        return (v + a * dt, v * dt + a * (0.5 * dt * dt));
    }
    let gravity = G * (1. - WATER_DENSITY / GOLD_DENSITY);
    let k = gravity / terminal_speed(diameter);
    let equilibrium = flow - DVec3::Y * (gravity / k);
    let decay = (-k * dt).exp();
    let response = -(-k * dt).exp_m1() / k;
    (
        equilibrium + (v - equilibrium) * decay,
        equilibrium * dt + (v - equilibrium) * response,
    )
}
fn mirror(p: &mut GoldParticle) {
    p.grain.position = p.position.as_vec3();
    p.grain.velocity = p.velocity.as_vec3();
    p.grain.resting = p.supported || p.buried;
}
fn contact_velocity(velocity: DVec3, normal: DVec3, mu: f64) -> DVec3 {
    let impulse = (-velocity.dot(normal)).max(0.);
    let tangent = velocity + normal * impulse;
    let speed = tangent.length();
    tangent * (1. - (mu * impulse / speed.max(1e-300)).min(1.))
}
fn forced_motion(v: DVec3, force_per_mass: DVec3, k: f64, dt: f64) -> (DVec3, DVec3) {
    if k == 0. {
        return (
            v + force_per_mass * dt,
            v * dt + force_per_mass * (0.5 * dt * dt),
        );
    }
    let equilibrium = force_per_mass / k;
    let response = -(-k * dt).exp_m1() / k;
    (
        equilibrium + (v - equilibrium) * (-k * dt).exp(),
        equilibrium * dt + (v - equilibrium) * response,
    )
}
/// Collinear planar slip has an exact Coulomb-plus-linear-drag solution,
/// including stopping and static friction. Changing two-dimensional slip
/// freezes its friction direction only for this bounded spatial substep.
fn supported_motion(
    v: DVec3,
    flow: DVec3,
    normal: DVec3,
    mu: f64,
    dt: f64,
    diameter: f64,
    wet: bool,
) -> Option<(DVec3, DVec3)> {
    let gravity = DVec3::new(
        0.,
        -G * if wet {
            1. - WATER_DENSITY / GOLD_DENSITY
        } else {
            1.
        },
        0.,
    );
    let k = if wet {
        -gravity.y / terminal_speed(diameter)
    } else {
        0.
    };
    let external = flow * k + gravity;
    let load = -external.dot(normal);
    if load <= 0. {
        return None;
    }
    let drive = external + normal * load;
    let tangent = v - normal * v.dot(normal);
    let speed = tangent.length();
    let friction = mu * load;
    let start_from_rest = |time: f64| {
        if drive.length() <= friction {
            (DVec3::ZERO, DVec3::ZERO)
        } else {
            forced_motion(DVec3::ZERO, drive - drive.normalize() * friction, k, time)
        }
    };
    if speed <= 1e-14 {
        return Some(start_from_rest(dt));
    }
    let direction = tangent / speed;
    let forcing = drive - direction * friction;
    let (velocity, displacement) = forced_motion(tangent, forcing, k, dt);
    if velocity.dot(direction) < 0. && drive.cross(direction).length() < 1e-12 {
        let opposing = -forcing.dot(direction);
        let stop = if k == 0. {
            speed / opposing
        } else {
            (k * speed / opposing).ln_1p() / k
        };
        let first = forced_motion(tangent, forcing, k, stop).1;
        let (last_velocity, last_displacement) = start_from_rest((dt - stop).max(0.));
        Some((last_velocity, first + last_displacement))
    } else {
        Some((velocity, displacement))
    }
}
fn advance(
    p: &mut GoldParticle,
    site: &Site,
    dt: f64,
    mu: f64,
    work: &mut usize,
) -> Result<bool, String> {
    if !in_claim(p.position) {
        return Ok(false);
    }
    let radius = p.grain.diameter_m as f64 * 0.5;
    let mut remaining = dt;
    let mut steps = 0;
    while remaining > dt * 1e-13 {
        steps += 1;
        *work += 1;
        if steps > MAX_SUBSTEPS || *work > MAX_WORK {
            return Err(
                "Independent gold numerical work capacity exceeded; timestep retained".into(),
            );
        }
        let (height, normal) = bed_at(site, p.position);
        let bottom = p.position.y - radius / normal.y;
        if p.buried {
            if height >= bottom - CONTACT_EPS {
                p.velocity = DVec3::ZERO;
                p.mobile = false;
                p.last_bed_height = Some(height);
                mirror(p);
                return Ok(true);
            }
            p.buried = false;
            p.supported = false;
        }
        let newly_covered =
            p.last_bed_height.is_some_and(|old| height > old + 1e-12) && bottom < height;
        if newly_covered
            || (bottom < height - CONTACT_EPS && (p.last_bed_height.is_none() || p.supported))
        {
            // Deposition covers the existing grain; never raise it to the surface.
            p.buried = true;
            p.supported = false;
            p.mobile = false;
            p.velocity = DVec3::ZERO;
            p.last_bed_height = Some(height);
            mirror(p);
            return Ok(true);
        }
        if p.supported && bottom > height + CONTACT_EPS {
            p.supported = false;
        }
        if (bottom - height).abs() <= CONTACT_EPS && p.velocity.dot(normal) <= 0. {
            if !p.supported {
                p.velocity = contact_velocity(p.velocity, normal, mu);
            }
            p.supported = true;
        }
        let (surface, flow) = water(site, p.position);
        let wet = p.position.y <= surface + 1e-12 && surface > height;
        let bound = p
            .velocity
            .length()
            .max(flow.length() + terminal_speed(p.grain.diameter_m as f64))
            + G * remaining;
        let mut h = remaining.min((CELL as f64 / 16.) / bound.max(1e-12));
        // Land exactly on a locally constant water surface before switching
        // phase, so fast airborne grains do not spend a whole frame underwater.
        if !wet && surface.is_finite() && p.position.y > surface {
            let disc = p.velocity.y * p.velocity.y + 2. * G * (p.position.y - surface);
            let crossing = if p.velocity.y <= 0. {
                2. * (p.position.y - surface) / (disc.sqrt() - p.velocity.y)
            } else {
                (p.velocity.y + disc.sqrt()) / G
            };
            if crossing > 1e-12 {
                h = h.min(crossing);
            }
        }
        if !h.is_finite() || h <= 0. || remaining - h == remaining {
            return Err("Unrepresentable independent gold substep".into());
        }
        let old_position = p.position;
        let old_velocity = p.velocity;
        let supported = if p.supported {
            supported_motion(
                old_velocity,
                flow,
                normal,
                mu,
                h,
                p.grain.diameter_m as f64,
                wet,
            )
        } else {
            None
        };
        let was_supported = supported.is_some();
        let (mut velocity, displacement) = supported
            .unwrap_or_else(|| motion(old_velocity, flow, h, p.grain.diameter_m as f64, wet));
        let mut position = old_position + displacement;
        if position.x < 0.
            || position.z < 0.
            || position.x >= EXTENT as f64
            || position.z >= EXTENT as f64
        {
            return Ok(false);
        }
        let (mut next_bed, mut next_normal) = bed_at(site, position);
        let gap = position.y - radius / next_normal.y - next_bed;
        if gap <= if was_supported { CONTACT_EPS } else { 0. } {
            if !was_supported && bottom > height + CONTACT_EPS {
                // Resolve first contact within this substep. The next loop
                // advances all remaining time with supported forces.
                let (mut lo, mut hi) = (0., h);
                for _ in 0..48 {
                    let time = (lo + hi) * 0.5;
                    let candidate = old_position
                        + motion(old_velocity, flow, time, p.grain.diameter_m as f64, wet).1;
                    let (bed, normal) = bed_at(site, candidate);
                    if candidate.y - radius / normal.y > bed {
                        lo = time;
                    } else {
                        hi = time;
                    }
                }
                h = hi;
                let at = motion(old_velocity, flow, h, p.grain.diameter_m as f64, wet);
                velocity = at.0;
                position = old_position + at.1;
                (next_bed, next_normal) = bed_at(site, position);
            }
            velocity = contact_velocity(velocity, next_normal, mu);
            position.y = next_bed + radius / next_normal.y;
            p.supported = true;
        } else {
            p.supported = false;
        }
        p.position = position;
        p.velocity = velocity;
        p.mobile = wet && !p.supported;
        p.last_bed_height = Some(bed_at(site, position).0);
        let angle = p.grain.angular_velocity.length() * h as f32;
        if angle > 0. {
            p.grain.rotation = (Quat::from_axis_angle(p.grain.angular_velocity.normalize(), angle)
                * p.grain.rotation)
                .normalize();
        }
        mirror(p);
        remaining -= h;
    }
    Ok(true)
}

fn cut_depths(_site: &Site, cuts: &[(usize, f32)]) -> std::collections::BTreeMap<usize, f64> {
    let mut out = std::collections::BTreeMap::<usize, f64>::new();
    for &(i, depth) in cuts {
        if i < WIDTH * WIDTH && depth.is_finite() && depth > 0. {
            let entry = out.entry(i).or_default();
            *entry += depth as f64;
        }
    }
    out
}
fn cut_contains(
    p: &GoldParticle,
    site: &Site,
    depths: &std::collections::BTreeMap<usize, f64>,
) -> bool {
    if p.mobile || p.delay_s > 0. {
        return false;
    }
    let gx = (p.position.x / CELL as f64).clamp(0., (WIDTH - 1) as f64);
    let gz = (p.position.z / CELL as f64).clamp(0., (WIDTH - 1) as f64);
    let x = (gx.floor() as usize).min(WIDTH - 2);
    let z = (gz.floor() as usize).min(WIDTH - 2);
    let fx = gx - x as f64;
    let fz = gz - z as f64;
    let ids = if fx + fz <= 1. {
        [
            (z * WIDTH + x, 1. - fx - fz),
            (z * WIDTH + x + 1, fx),
            ((z + 1) * WIDTH + x, fz),
        ]
    } else {
        [
            ((z + 1) * WIDTH + x + 1, fx + fz - 1.),
            ((z + 1) * WIDTH + x, 1. - fx),
            (z * WIDTH + x + 1, 1. - fz),
        ]
    };
    let requested = ids
        .iter()
        .map(|(i, w)| depths.get(i).copied().unwrap_or(0.) * w)
        .sum::<f64>();
    if requested <= 0. {
        return false;
    }
    let depth = ids
        .iter()
        .map(|(i, w)| {
            let available = (0..4)
                .map(|class| site.bed_class_volume_m3(*i, class))
                .sum::<f64>()
                / (CELL as f64).powi(2);
            depths.get(i).copied().unwrap_or(0.).min(available) * w
        })
        .sum::<f64>();
    let (old, normal) = bed_at(site, p.position);
    let radius = p.grain.diameter_m as f64 * 0.5 / normal.y;
    let surface_overlap =
        p.position.y - radius <= old + CONTACT_EPS && p.position.y + radius >= old - CONTACT_EPS;
    surface_overlap
        || (depth > 0.
            && p.position.y - radius <= old + CONTACT_EPS
            && p.position.y + radius > old - depth)
}
