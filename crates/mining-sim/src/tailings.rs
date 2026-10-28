//! Finite sluice-solid discharge. Airborne grains retain world motion until
//! their first terrain/water contact. Minerals enter terrain; gold-bearing
//! packets retain explicit contact custody until a free-gold model exists.
//! Jet/air drag, granular runout and impact momentum coupling remain unmodeled.
use crate::{
    site::{CELL, EXTENT, WIDTH},
    sluice::Grain,
    Material, Site, Sluice, Vec3,
};
use glam::{DVec3, Mat3, Quat};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Numerical storage guard, not a physical feed/recovery limit. Refusal leaves
/// the entire removal batch in the solver and makes the shift saveable.
pub const MAX_PACKETS: usize = 65_536;
const G: f64 = 9.81;
const AREA: f64 = (CELL as f64) * (CELL as f64);
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TailingsPacket {
    pub id: u64,
    pub grain: Grain,
    /// First contact captured, but gold dynamics are not admitted to terrain.
    pub contact_pending: bool,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Tailings {
    pub packets: Vec<TailingsPacket>,
    pub released_count: u64,
    /// Cumulative diagnostics below are never current stored inventory.
    pub released: Material,
    pub received: Material,
    pub historic_bed: Material,
    /// Actual material which crossed the claim boundary; counted once.
    pub exported: Material,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct SurfaceReceipt {
    pub bed: Material,
    pub suspension: Material,
}
impl Tailings {
    pub fn active_material(&self) -> Material {
        self.packets.iter().fold(Material::default(), |mut a, p| {
            a.add(p.grain.material);
            a
        })
    }
    pub fn prepare_batch(&mut self, count: usize) -> Result<(), String> {
        if self
            .packets
            .len()
            .checked_add(count)
            .is_none_or(|n| n > MAX_PACKETS)
            || self.released_count.checked_add(count as u64).is_none()
        {
            return Err("Tailings flight storage full; outgoing solids retained in sluice".into());
        }
        self.packets.try_reserve(count).map_err(|_| {
            "Cannot allocate tailings flight storage; outgoing solids retained".to_string()
        })
    }
    /// Borrow the donor until acceptance succeeds. A failed individual
    /// admission cannot consume its inventory; batches preflight all entries.
    pub fn accept(&mut self, grain: &Grain) -> Result<(), String> {
        if !valid_grain(grain) {
            return Err("Invalid outgoing solid; donor retained".into());
        }
        self.prepare_batch(1)?;
        self.released_count += 1;
        self.released.add(grain.material);
        self.packets.push(TailingsPacket {
            id: self.released_count,
            grain: grain.clone(),
            contact_pending: false,
        });
        Ok(())
    }
    pub fn valid(&self) -> bool {
        let mut ids = BTreeSet::new();
        if self.packets.len() > MAX_PACKETS
            || ![
                self.released,
                self.received,
                self.historic_bed,
                self.exported,
            ]
            .iter()
            .all(|m| m.valid())
            || self.packets.iter().any(|p| {
                p.id == 0
                    || p.id > self.released_count
                    || !ids.insert(p.id)
                    || (p.contact_pending && p.grain.material.gold_g == 0.)
                    || !valid_grain(&p.grain)
            })
        {
            return false;
        }
        let mut accounted = self.active_material();
        accounted.add(self.received);
        accounted.add(self.exported);
        fields(accounted)
            .into_iter()
            .zip(fields(self.released))
            .all(|(a, b)| (a - b).abs() <= (b * 1e-9).max(1e-15))
            && fields(self.historic_bed)
                .into_iter()
                .zip(fields(self.received))
                .all(|(a, b)| a <= b + (b * 1e-9).max(1e-15))
    }
    /// Return just this step's boundary export for the external water ledger.
    pub fn step(&mut self, site: &mut Site, dt: f32) -> Material {
        assert!(dt.is_finite() && (0.0..=0.25).contains(&dt));
        if dt == 0. {
            return Material::default();
        }
        let mut received = Material::default();
        let mut exported = Material::default();
        self.packets.retain_mut(|p| {
            if p.contact_pending {
                if admit_gold(site, &p.grain, dt) {
                    received.add(p.grain.material);
                    return false;
                }
                return true;
            }
            let g = &mut p.grain;
            let end = first_event(site, g, dt as f64);
            if let Some((time, boundary)) = end {
                let at = trajectory(g, time).as_vec3();
                if boundary {
                    exported.add(g.material);
                } else if g.material.gold_g > 0. {
                    // The terrain gold store is a class-3 provenance tag, not
                    // an independent free-gold phase. Retain the complete
                    // packet at first contact rather than strand or relabel it.
                    g.position = at;
                    g.velocity.y -= (G * time) as f32;
                    let angle = g.angular_velocity.length() * time as f32;
                    if angle > 0. {
                        g.rotation = (Quat::from_axis_angle(g.angular_velocity.normalize(), angle)
                            * g.rotation)
                            .normalize();
                    }
                    p.contact_pending = true;
                    if admit_gold(site, g, (dt as f64 - time).max(0.) as f32) {
                        received.add(g.material);
                        return false;
                    }
                    return true;
                } else {
                    // Valid admitted world coordinates and material; no fallible
                    // work remains after material leaves this flight owner.
                    site.receive_tailings(at, g.material, false)
                        .expect("admitted tailings impact");
                    received.add(g.material);
                }
                false
            } else {
                g.position = trajectory(g, dt as f64).as_vec3();
                g.velocity.y -= (G * dt as f64) as f32;
                let angle = g.angular_velocity.length() * dt;
                if angle > 0. {
                    g.rotation = (Quat::from_axis_angle(g.angular_velocity.normalize(), angle)
                        * g.rotation)
                        .normalize();
                }
                true
            }
        });
        self.received.add(received);
        self.exported.add(exported);
        exported
    }
}
fn fields(m: Material) -> [f64; 5] {
    [m.soil_m3, m.sand_m3, m.gravel_m3, m.gold_g, m.pore_water_m3]
}
pub(crate) fn valid_grain(g: &Grain) -> bool {
    g.position.is_finite()
        && g.position.length() < 1000.
        && g.velocity.is_finite()
        && g.velocity.length() < 200.
        && g.affine.is_finite()
        && g.rotation.is_finite()
        && (g.rotation.length_squared() - 1.).abs() < 1e-3
        && g.angular_velocity.is_finite()
        && g.material.valid()
        && g.solid_volume() > 0.
        && (0.00002..=0.02).contains(&g.diameter_m)
        && g.collision_radius().is_finite()
        && !g.resting
}
/// Grid-to-world has a reflected downstream/up/right basis. Polar vectors and
/// affine gradients use B; axial angular velocity uses det(B) B = -B.
pub fn world_grain(s: &Sluice, g: &Grain) -> Result<Grain, String> {
    let p = s
        .placement
        .ok_or("Unplaced sluice cannot discharge spatial solids")?;
    let b = Mat3::from_cols(p.forward(), Vec3::Y, p.right());
    let mut out = g.clone();
    out.position = p.to_world(s.grid_to_local(g.position));
    out.velocity = b * g.velocity;
    out.affine = b * g.affine * b.transpose();
    out.angular_velocity = -(b * g.angular_velocity);
    out.rotation = Quat::from_mat3(&(b * Mat3::from_quat(g.rotation) * b.transpose())).normalize();
    out.resting = false;
    if !valid_grain(&out) {
        return Err("Invalid outgoing solid; retained in sluice".into());
    }
    Ok(out)
}
/// Inverse of the reflected grid/world boundary. No re-emission, position
/// clamp, added impulse, grading change or loss of axial/affine state.
pub fn grid_grain(s: &Sluice, g: &Grain) -> Result<Grain, String> {
    let p = s.placement.ok_or("Place the receiving sluice first")?;
    if !valid_grain(g) {
        return Err("Invalid world solid; retained by plant".into());
    }
    let b = Mat3::from_cols(p.forward(), Vec3::Y, p.right());
    let local = b.transpose() * (g.position - p.inlet);
    let floor = s.grid_to_local(Vec3::ZERO).y;
    let mut out = g.clone();
    out.position = Vec3::new(
        local.x,
        local.y - floor,
        local.z + s.profile.width() * 0.5 + s.profile.cell(),
    );
    out.velocity = b.transpose() * g.velocity;
    out.affine = b.transpose() * g.affine * b;
    out.angular_velocity = -(b.transpose() * g.angular_velocity);
    out.rotation = Quat::from_mat3(&(b.transpose() * Mat3::from_quat(g.rotation) * b)).normalize();
    Ok(out)
}
fn admit_gold(site: &mut Site, g: &Grain, dt: f32) -> bool {
    let Some(mut gold) = site.loose_gold.take() else {
        return false;
    };
    let result = gold.admit_and_advance(g, site, dt);
    site.loose_gold = Some(gold);
    result.is_ok()
}
fn pure_gold(g: &Grain) -> bool {
    g.kind == crate::GrainKind::Gold
        && g.material.gold_g > 0.
        && g.material.volume() == 0.
        && g.material.pore_water_m3 == 0.
}
fn trajectory(g: &Grain, t: f64) -> DVec3 {
    g.position.as_dvec3() + g.velocity.as_dvec3() * t - DVec3::Y * (0.5 * G * t * t)
}
fn bilinear(site: &Site, water: bool, p: DVec3) -> f64 {
    let x = (p.x / CELL as f64).clamp(0., (WIDTH - 1) as f64);
    let z = (p.z / CELL as f64).clamp(0., (WIDTH - 1) as f64);
    let ix = x.floor() as usize;
    let iz = z.floor() as usize;
    let nx = (ix + 1).min(WIDTH - 1);
    let nz = (iz + 1).min(WIDTH - 1);
    let h = |a, b| {
        if water {
            site.world.water_surface[b * WIDTH + a] as f64
        } else {
            site.world.ground_height(a, b) as f64
        }
    };
    let a = h(ix, iz) * (1. - x.fract()) + h(nx, iz) * x.fract();
    let b = h(ix, nz) * (1. - x.fract()) + h(nx, nz) * x.fract();
    a * (1. - z.fract()) + b * z.fract()
}
/// Earliest zero of a quadratic on [0,1], including an interior excursion even
/// if both endpoints are clear. Stable roots avoid cancellation at small dt.
fn crossing(c: f64, m: f64, e: f64) -> Option<f64> {
    if c <= 0. {
        return Some(0.);
    }
    let a = 2. * (e + c - 2. * m);
    let b = e - c - a;
    if a.abs() <= 1e-12 * (c.abs() + m.abs() + e.abs()).max(1.) {
        return (b < 0.).then(|| -c / b).filter(|u| (0.0..=1.0).contains(u));
    }
    let d = b * b - 4. * a * c;
    if d < 0. {
        return None;
    }
    let q = -0.5 * (b + if b >= 0. { d.sqrt() } else { -d.sqrt() });
    [q / a, if q != 0. { c / q } else { f64::INFINITY }]
        .into_iter()
        .filter(|u| u.is_finite() && (0.0..=1.0).contains(u))
        .min_by(f64::total_cmp)
}
fn first_event(site: &Site, g: &Grain, dt: f64) -> Option<(f64, bool)> {
    let p = g.position.as_dvec3();
    let v = g.velocity.as_dvec3();
    if !(0.0..=EXTENT as f64).contains(&p.x) || !(0.0..=EXTENT as f64).contains(&p.z) {
        return Some((0., true));
    }
    let mut boundary = dt + 1.;
    for (x, speed) in [(p.x, v.x), (p.z, v.z)] {
        if speed > 0. {
            boundary = boundary.min((EXTENT as f64 - x) / speed);
        } else if speed < 0. {
            boundary = boundary.min(-x / speed);
        }
    }
    let stop = dt.min(boundary);
    let mut breaks = vec![0., stop];
    for (x, speed) in [(p.x, v.x), (p.z, v.z)] {
        if speed == 0. {
            continue;
        }
        let lo = x.min(x + speed * stop);
        let hi = x.max(x + speed * stop);
        for k in ((lo / CELL as f64).floor() as i32 + 1)..=((hi / CELL as f64).ceil() as i32) {
            let t = (k as f64 * CELL as f64 - x) / speed;
            if t > 0. && t < stop {
                breaks.push(t);
            }
        }
    }
    // Pure gold uses the same triangle contact surface as its downstream owner.
    // Include diagonal crossings so each interval has one affine bed plane.
    if pure_gold(g) && v.x + v.z != 0. {
        let a = p.x + p.z;
        let speed = v.x + v.z;
        let lo = a.min(a + speed * stop);
        let hi = a.max(a + speed * stop);
        for k in ((lo / CELL as f64).floor() as i32 + 1)..=((hi / CELL as f64).ceil() as i32) {
            let time = (k as f64 * CELL as f64 - a) / speed;
            if time > 0. && time < stop {
                breaks.push(time);
            }
        }
    }
    breaks.sort_by(f64::total_cmp);
    breaks.dedup();
    for interval in breaks.windows(2) {
        let (a, b) = (interval[0], interval[1]);
        let mid = (a + b) * 0.5;
        let mut hit = None;
        for water in [false, true] {
            let gap = |t| {
                let at = trajectory(g, t);
                if !water && pure_gold(g) {
                    let (height, normal) = crate::loose_gold::bed_at(site, at);
                    at.y - g.diameter_m as f64 * 0.5 / normal.y - height
                } else {
                    let radius = if pure_gold(g) {
                        g.diameter_m * 0.5
                    } else {
                        g.collision_radius()
                    };
                    at.y - radius as f64 - bilinear(site, water, at)
                }
            };
            if let Some(u) = crossing(gap(a), gap(mid), gap(b)) {
                let t = a + (b - a) * u;
                hit = Some(hit.map_or(t, |h: f64| h.min(t)));
            }
        }
        if let Some(t) = hit {
            return Some((t, false));
        }
    }
    (boundary <= dt).then_some((boundary, true))
}
impl Site {
    /// Bilinear receiving footprint, bounded to the actual contact location.
    /// Historic pooled stock is reconstructed as a bed at its former visible
    /// pile location. New water impacts enter existing terrain size classes.
    pub fn receive_tailings(
        &mut self,
        p: Vec3,
        m: Material,
        force_bed: bool,
    ) -> Result<SurfaceReceipt, String> {
        if !p.is_finite()
            || p.x < 0.
            || p.z < 0.
            || p.x > EXTENT
            || p.z > EXTENT
            || !m.valid()
            || m.gold_g > 0.
        {
            return Err("Invalid tailings receiver or unsupported free-gold handoff".into());
        }
        let x = (p.x as f64 / CELL as f64).clamp(0., (WIDTH - 1) as f64);
        let z = (p.z as f64 / CELL as f64).clamp(0., (WIDTH - 1) as f64);
        let ix = x.floor() as usize;
        let iz = z.floor() as usize;
        let weights = [
            (1. - x.fract()) * (1. - z.fract()),
            x.fract() * (1. - z.fract()),
            (1. - x.fract()) * z.fract(),
            x.fract() * z.fract(),
        ];
        let cells = [
            iz * WIDTH + ix,
            iz * WIDTH + (ix + 1).min(WIDTH - 1),
            (iz + 1).min(WIDTH - 1) * WIDTH + ix,
            (iz + 1).min(WIDTH - 1) * WIDTH + (ix + 1).min(WIDTH - 1),
        ];
        let mut receipt = SurfaceReceipt::default();
        for (i, weight) in cells.into_iter().zip(weights) {
            if weight == 0. {
                continue;
            }
            let portion = m.scaled(weight);
            let depth = self.world.water_depth(i % WIDTH, i / WIDTH) as f64;
            if !force_bed && depth > 0. {
                let mut inventory = self.suspended_class_volumes_m3(i);
                for (value, amount) in inventory.iter_mut().zip([
                    portion.sand_m3,
                    portion.soil_m3,
                    portion.gravel_m3,
                    0.,
                ]) {
                    *value += amount;
                }
                self.set_suspension_inventory_m3(i, inventory);
                if portion.gold_g > 0. {
                    if self.hydraulic.suspended_gold_g.is_empty() {
                        self.hydraulic.suspended_gold_g = vec![0.; WIDTH * WIDTH];
                    }
                    self.hydraulic.suspended_gold_g[i] += portion.gold_g;
                }
                self.receive_surface_water(i, portion.pore_water_m3, [0.; 2]);
                receipt.suspension.add(portion);
            } else {
                let inventory = self.suspended_class_volumes_m3(i);
                self.set_suspension_inventory_m3(i, [0.; 4]);
                let volume = depth * AREA;
                let momentum = std::array::from_fn(|axis| {
                    volume
                        * if axis == 0 {
                            self.hydraulic.velocity_x.get(i)
                        } else {
                            self.hydraulic.velocity_z.get(i)
                        }
                        .copied()
                        .unwrap_or(0.) as f64
                });
                for (class, amount) in [
                    (0, portion.sand_m3),
                    (1, portion.soil_m3),
                    (2, portion.gravel_m3),
                ] {
                    self.add_bed_volume_m3(i, class, amount);
                }
                self.gold_g[i] += portion.gold_g;
                self.world.water_surface[i] = self.world.ground_height(i % WIDTH, i / WIDTH);
                self.receive_surface_water(i, volume, momentum);
                self.set_suspension_inventory_m3(i, inventory);
                self.add_pore_water(i, portion.pore_water_m3);
                receipt.bed.add(portion);
            }
        }
        self.hydraulic.impact_dirty = true;
        self.revision += 1;
        Ok(receipt)
    }
}
