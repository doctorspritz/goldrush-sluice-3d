//! Statistical bulk-deck approximation, deliberately separate from wire DEM.
//! Finite weighted parcels own all stock. The coarse bed only derives depth and
//! adhesive coverage; it never creates inventory. Passage is a saved deterministic
//! residence hazard, not a recovery bonus. Parameters remain uncalibrated.
use super::{rotate_grain, Screen, ScreenParticle, G};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BulkDeck {
    pub cells_downstream: u16,
    pub cells_across: u16,
    /// Assumed yield stress of saturated soil fines; not a measured clay law.
    pub soil_yield_pa: f64,
}
impl Default for BulkDeck {
    fn default() -> Self {
        Self {
            cells_downstream: 16,
            cells_across: 4,
            soil_yield_pa: 25.,
        }
    }
}
impl BulkDeck {
    pub(super) fn valid(&self) -> bool {
        (1..=64).contains(&self.cells_downstream)
            && (1..=16).contains(&self.cells_across)
            && (0.0..=200.).contains(&self.soil_yield_pa)
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Residence {
    on_deck: bool,
    exposure: f64,
    seconds: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct BulkState {
    pub(super) config: BulkDeck,
    residence: BTreeMap<u64, Residence>,
}
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Cell {
    pub(super) depth_m: f64,
    pub(super) blinded: f64,
}
impl BulkState {
    pub(super) fn new(config: BulkDeck) -> Self {
        Self {
            config,
            residence: BTreeMap::new(),
        }
    }
    pub(super) fn admit(&mut self, id: u64) {
        self.residence.insert(id, Residence::default());
    }
    pub(super) fn clear(&mut self) {
        self.residence.clear();
    }
    pub(super) fn valid(&self, screen: &Screen) -> bool {
        self.config.valid()
            && self.residence.len() == screen.particles.len()
            && screen.particles.iter().all(|p| {
                self.residence.get(&p.id).is_some_and(|r| {
                    r.exposure.is_finite()
                        && r.exposure >= 0.
                        && r.seconds.is_finite()
                        && r.seconds >= 0.
                        && !(r.on_deck && p.passed_deck)
                })
            })
    }
    fn cell(&self, screen: &Screen, local: DVec3) -> usize {
        let c = &screen.config;
        let nx = self.config.cells_downstream as usize;
        let nz = self.config.cells_across as usize;
        let x = (local.x / c.length_m * nx as f64)
            .floor()
            .clamp(0., (nx - 1) as f64) as usize;
        let z = ((local.z / c.width_m + 0.5) * nz as f64)
            .floor()
            .clamp(0., (nz - 1) as f64) as usize;
        x * nz + z
    }
    fn adhesion(&self, p: &ScreenParticle) -> f64 {
        let m = p.grain.material;
        let density = m.mass_kg() / m.solid_volume_m3().max(1e-30);
        let d = p.grain.diameter_m as f64;
        let soil = (m.soil_m3 / m.volume().max(1e-30)).clamp(0., 1.);
        let saturation = m.saturation();
        // Pendular bridges weaken again at saturation. Cohesive soil remains a
        // distinct yield-stress approximation. No surrounding deck-water flow
        // or spray washing is inferred from stored free-water credits.
        let bridges = 4. * saturation * (1. - saturation);
        6. * 0.072 * bridges * (1. - soil) / (density * d * d)
            + 1.5 * self.config.soil_yield_pa * soil * saturation / (density * d)
    }
    fn shaking(&self, screen: &Screen) -> f64 {
        if screen.motor_on {
            screen.config.amplitude_normal_m
                * (std::f64::consts::TAU * screen.config.frequency_hz).powi(2)
        } else {
            0.
        }
    }
    pub(super) fn bed(&self, screen: &Screen) -> Vec<Cell> {
        self.bed_at(screen, screen.time)
    }
    fn bed_at(&self, screen: &Screen, time: f64) -> Vec<Cell> {
        let mut cells = vec![
            Cell::default();
            self.config.cells_downstream as usize
                * self.config.cells_across as usize
        ];
        let area = screen.config.length_m
            / self.config.cells_downstream as f64
            / screen.config.slope_deg.to_radians().cos()
            * screen.config.width_m
            / self.config.cells_across as f64;
        let drive = self.shaking(screen);
        let offset = screen.offset_at(time).0;
        for p in &screen.particles {
            if !self.residence.get(&p.id).is_some_and(|r| r.on_deck) {
                continue;
            }
            let index = self.cell(screen, screen.to_local(p.position - offset));
            let cell = &mut cells[index];
            cell.depth_m += p.grain.material.handling_volume_m3() / area;
            if self.adhesion(p) > drive {
                // Projected finite solid area of a representative sphere cohort:
                // N*pi*d^2/4 = 1.5*V_solid/d. No extra 'plug' mass is minted.
                cell.blinded +=
                    1.5 * p.grain.material.solid_volume_m3() / (p.grain.diameter_m as f64 * area);
            }
        }
        for c in &mut cells {
            c.blinded = c.blinded.clamp(0., 1.);
        }
        cells
    }
    pub(super) fn step(&mut self, screen: &mut Screen, dt: f64) -> Result<(), String> {
        let count = (dt * 240.).ceil().max(1.) as usize;
        let h = dt / count as f64;
        for substep in 0..count {
            let at = substep as f64 * h;
            let t0 = screen.drive_time(at);
            let t1 = screen.drive_time(at + h);
            let (off0, _) = screen.offset_at(t0);
            let (off1, deck_velocity) = screen.offset_at(t1);
            let cells = self.bed_at(screen, t0);
            let mut particles = std::mem::take(&mut screen.particles);
            for mut p in particles.drain(..) {
                screen.last_step.intervals += 1;
                let mut residence = self
                    .residence
                    .remove(&p.id)
                    .ok_or("Missing bulk parcel residence")?;
                let was_on_deck = residence.on_deck;
                let r = p.grain.diameter_m as f64 * 0.5;
                let mut local = screen.to_local(p.position - off0);
                let inside = local.x >= 0.
                    && local.x <= screen.config.length_m
                    && local.z.abs() <= screen.config.width_m * 0.5;
                let index = self.cell(screen, local);
                let cell = cells[index];
                let normal = screen.normal();
                let target = r + 2. * screen.config.wire_radius_m + 0.5 * cell.depth_m;
                let mut landed = false;
                if !p.passed_deck && !residence.on_deck {
                    p.position += p.velocity * h - DVec3::Y * (0.5 * G * h * h);
                    p.velocity.y -= G * h;
                    local = screen.to_local(p.position - off1);
                    if local.x >= 0.
                        && local.x <= screen.config.length_m
                        && local.z.abs() <= screen.config.width_m * 0.5
                        && screen.deck_distance(local) <= target
                    {
                        residence.on_deck = true;
                        landed = true;
                    }
                }
                if residence.on_deck && !inside && !landed {
                    residence.on_deck = false;
                }
                if residence.on_deck {
                    let drive = self.shaking(screen);
                    let gamma = drive / (G * screen.config.slope_deg.to_radians().cos());
                    let adhesive = self.adhesion(&p);
                    let mu = screen.config.friction_mu / (1. + gamma);
                    let mut relative = p.velocity - deck_velocity;
                    relative -= normal * relative.dot(normal);
                    // Depth gradients cause local load to spread/overflow instead
                    // of a width-wide capacity flag. Coulomb friction resists it.
                    let nx = self.config.cells_downstream as usize;
                    let nz = self.config.cells_across as usize;
                    let ix = index / nz;
                    let iz = index % nz;
                    let x0 = ix.saturating_sub(1);
                    let x1 = (ix + 1).min(nx - 1);
                    let z0 = iz.saturating_sub(1);
                    let z1 = (iz + 1).min(nz - 1);
                    let gx = (cells[x1 * nz + iz].depth_m - cells[x0 * nz + iz].depth_m)
                        / ((x1 - x0).max(1) as f64 * screen.config.length_m / nx as f64);
                    let gz = (cells[ix * nz + z1].depth_m - cells[ix * nz + z0].depth_m)
                        / ((z1 - z0).max(1) as f64 * screen.config.width_m / nz as f64);
                    let gravity = -DVec3::Y * G;
                    let mut acceleration = gravity
                        - normal * gravity.dot(normal)
                        - G * (screen.downstream() * gx + screen.across() * gz);
                    acceleration -= normal * acceleration.dot(normal);
                    relative += acceleration * h;
                    let friction = mu * G * screen.config.slope_deg.to_radians().cos() + adhesive;
                    let speed = relative.length();
                    relative *= (1. - friction * h / speed.max(1e-30)).max(0.);
                    p.position += (off1 - off0) + relative * h;
                    p.velocity = relative + deck_velocity;
                    local = screen.to_local(p.position - off1);
                    let correction = target - screen.deck_distance(local);
                    p.position += normal * correction;
                    screen.last_step.projection_work_j +=
                        p.grain.material.wet_mass_kg() * G * correction * normal.y;
                    screen.last_step.maximum_repair_m =
                        screen.last_step.maximum_repair_m.max(correction.abs());
                    residence.seconds += h;
                    let d = p.grain.diameter_m as f64;
                    let clearance = (screen.config.aperture_m - d).max(0.);
                    let probability =
                        (clearance / screen.config.pitch()).powi(2) * (1. - cell.blinded);
                    let access = d / (d + cell.depth_m);
                    let attempts = if adhesive <= drive {
                        screen.config.frequency_hz
                            * if screen.motor_on {
                                gamma / (1. + gamma)
                            } else {
                                0.
                            }
                            + relative.length() / screen.config.pitch()
                    } else {
                        0.
                    };
                    residence.exposure += -(-probability).ln_1p()
                        * access
                        * (attempts * h + if landed && adhesive <= drive { 1. } else { 0. });
                    if residence.exposure >= passage_threshold(p.id) && probability > 0. {
                        // Unresolved transit through the bulk/mesh. Preserve the
                        // parcel's grading, spin and complete finite material.
                        p.position -= normal * (target + r + 2. * screen.config.wire_radius_m);
                        p.passed_deck = true;
                        residence.on_deck = false;
                    }
                } else if p.passed_deck || was_on_deck {
                    p.position += p.velocity * h - DVec3::Y * (0.5 * G * h * h);
                    p.velocity.y -= G * h;
                }
                let local = screen.to_local(p.position);
                let half = screen.config.width_m * 0.5;
                if p.passed_deck
                    && local.x >= 0.
                    && local.x <= screen.config.length_m + r
                    && local.z.abs() <= half
                {
                    let floor = -0.15 - 0.40 * local.x / screen.config.length_m
                        + 0.08 * local.z.abs() / half;
                    let n = (screen.downstream() * (0.40 / screen.config.length_m) + DVec3::Y
                        - screen.across() * (0.08 / half * local.z.signum()))
                    .normalize();
                    if local.y <= floor + r / n.y {
                        p.position.y = screen.origin.y as f64 + floor + r / n.y;
                        p.velocity -= n * p.velocity.dot(n).min(0.);
                        let tangent = p.velocity - n * p.velocity.dot(n);
                        let speed = tangent.length();
                        p.velocity -= tangent
                            * (screen.config.friction_mu * G * n.y * h / speed.max(1e-30)).min(1.);
                        p.in_gutter = true;
                    }
                }
                p.grain.position = p.position.as_vec3();
                p.grain.velocity = p.velocity.as_vec3();
                p.grain.resting = residence.on_deck && !screen.motor_on;
                rotate_grain(&mut p, h);
                if !super::valid_grain(&p.grain)
                    || !p.position.is_finite()
                    || !p.velocity.is_finite()
                {
                    return Err(
                        "Bulk screen motion exceeded supported bounds; whole step retained".into(),
                    );
                }
                let local = screen.to_local(p.position);
                let bin_floor =
                    -screen.config.length_m * screen.config.slope_deg.to_radians().tan() - 0.10;
                if p.in_gutter && local.x >= screen.config.length_m + r {
                    screen.undersize.push(p.grain);
                } else if !p.passed_deck
                    && local.x >= screen.config.length_m
                    && local.x <= screen.config.length_m + 0.18
                    && local.z.abs() <= half
                    && local.y <= bin_floor + r
                {
                    p.grain.velocity = crate::Vec3::ZERO;
                    screen.oversize.push(p.grain);
                } else if local.x < -r - 0.02
                    || local.x > screen.config.length_m + 0.18 + r
                    || local.z.abs() > half + r
                    || local.y < -0.8
                {
                    screen.spills.push(p.grain);
                } else {
                    self.residence.insert(p.id, residence);
                    screen.particles.push(p);
                }
            }
        }
        Ok(())
    }
}
fn passage_threshold(id: u64) -> f64 {
    // Counter-based deterministic sample, saved parcel ID; no process RNG or
    // extra attempt on save/load. Exponential threshold integrates hazard.
    let mut x = id.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    -(((x >> 11) as f64 + 0.5) / ((1_u64 << 53) as f64)).ln()
}
