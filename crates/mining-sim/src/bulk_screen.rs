//! Statistical bulk-deck approximation, deliberately separate from wire DEM.
//! Finite weighted parcels own all stock. The coarse bed only derives depth and
//! adhesive coverage; it never creates inventory. Passage is a saved deterministic
//! residence hazard, not a recovery bonus. Parameters remain uncalibrated.
use super::{rotate_grain, BulkBedField, BulkProjection, Screen, ScreenParticle, G};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BulkDeck {
    pub cells_downstream: u16,
    pub cells_across: u16,
    /// Assumed yield stress of saturated soil fines; not a measured clay law.
    pub soil_yield_pa: f64,
    /// Missing in saved model 2; never silently migrate its force law.
    #[serde(default)]
    pub projection: BulkProjection,
}
impl Default for BulkDeck {
    fn default() -> Self {
        Self {
            cells_downstream: 16,
            cells_across: 4,
            soil_yield_pa: 25.,
            projection: BulkProjection::SmoothCellsV2,
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
    fn field_coordinates(&self, screen: &Screen, local: DVec3) -> glam::DVec2 {
        if self.config.projection == BulkProjection::CellBinsV1 {
            glam::DVec2::new(local.x, local.z)
        } else {
            let angle = screen.config.slope_deg.to_radians();
            // Project onto the deck tangent before lookup, so a normal lift
            // cannot spuriously move a parcel into a neighbouring bed cell.
            glam::DVec2::new(
                (local.x * angle.cos() - local.y * angle.sin()) * angle.cos(),
                local.z,
            )
        }
    }
    fn inside(&self, screen: &Screen, point: glam::DVec2) -> bool {
        point.x >= 0.
            && point.x <= screen.config.length_m
            && point.y.abs() <= screen.config.width_m * 0.5
    }
    pub(super) fn supported_ids(&self, screen: &Screen) -> Vec<u64> {
        let offset = screen.offset_at(screen.time).0;
        screen
            .particles
            .iter()
            .filter(|p| {
                self.residence.get(&p.id).is_some_and(|r| r.on_deck)
                    && (self.config.projection == BulkProjection::CellBinsV1
                        || self.inside(
                            screen,
                            self.field_coordinates(screen, screen.to_local(p.position - offset)),
                        ))
            })
            .map(|p| p.id)
            .collect()
    }
    fn adhesion(&self, p: &ScreenParticle) -> f64 {
        let m = p.grain.material;
        let inertial_mass = if self.config.projection == BulkProjection::CellBinsV1 {
            m.mass_kg()
        } else {
            m.wet_mass_kg()
        };
        let density = inertial_mass / m.solid_volume_m3().max(1e-30);
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
    fn transport_adhesion(&self, p: &ScreenParticle, depth: f64) -> f64 {
        let acceleration = self.adhesion(p);
        if self.config.projection == BulkProjection::CellBinsV1 {
            acceleration
        } else {
            // Basal traction acts on the bottom contact layer, not on every
            // internal bond in the bed. Deep-sheet force then scales with area
            // rather than growing without bound with the number of layers.
            let diameter = p.grain.diameter_m as f64;
            acceleration * diameter / (diameter + depth)
        }
    }
    fn shaking(&self, screen: &Screen) -> f64 {
        if screen.motor_on {
            screen.config.amplitude_normal_m
                * (std::f64::consts::TAU * screen.config.frequency_hz).powi(2)
        } else {
            0.
        }
    }
    pub(super) fn field_at(&self, screen: &Screen, time: f64) -> BulkBedField {
        let mut field = BulkBedField::new(screen, &self.config);
        let area = field.cell_area();
        let drive = self.shaking(screen);
        let offset = screen.offset_at(time).0;
        for p in &screen.particles {
            if !self.residence.get(&p.id).is_some_and(|r| r.on_deck) {
                continue;
            }
            let point = self.field_coordinates(screen, screen.to_local(p.position - offset));
            if self.config.projection != BulkProjection::CellBinsV1 && !self.inside(screen, point) {
                continue;
            }
            // Keep the original coverage arithmetic order for saved model 2.
            let coverage = if self.adhesion(p) > drive {
                1.5 * p.grain.material.solid_volume_m3() / (p.grain.diameter_m as f64 * area)
            } else {
                0.
            };
            field.deposit(point.x, point.y, p.grain.material, coverage);
        }
        field.finish();
        field
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
            let field = self.field_at(screen, t0);
            let cells = field.cells();
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
                let point = self.field_coordinates(screen, local);
                let inside = self.inside(screen, point);
                let index = field.index(point.x, point.y);
                let cell = if self.config.projection == BulkProjection::CellBinsV1 {
                    cells[index]
                } else {
                    let sample = field.sample(point.x, point.y);
                    super::BulkBedCell {
                        material: sample.material,
                        depth_m: sample.depth_m,
                        blinded: sample.blinded,
                    }
                };
                let normal = screen.normal();
                let target = r + 2. * screen.config.wire_radius_m + 0.5 * cell.depth_m;
                let mut landed = false;
                if !p.passed_deck && !residence.on_deck {
                    p.position += p.velocity * h - DVec3::Y * (0.5 * G * h * h);
                    p.velocity.y -= G * h;
                    local = screen.to_local(p.position - off1);
                    if self.inside(screen, self.field_coordinates(screen, local))
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
                    let mut acceleration = if self.config.projection == BulkProjection::CellBinsV1 {
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
                        gravity
                            - normal * gravity.dot(normal)
                            - G * (screen.downstream() * gx + screen.across() * gz)
                    } else {
                        let angle = screen.config.slope_deg.to_radians();
                        let tangent = screen.downstream() * angle.cos() - DVec3::Y * angle.sin();
                        let gravity = -DVec3::Y * G;
                        let slopes = if landed {
                            glam::DVec2::ZERO
                        } else {
                            field.pressure_slopes(point.x, point.y, p.grain.material)
                        };
                        gravity
                            - normal * gravity.dot(normal)
                            - G * angle.cos()
                                * (tangent * (slopes.x * angle.cos()) + screen.across() * slopes.y)
                    };
                    acceleration -= normal * acceleration.dot(normal);
                    relative += acceleration * h;
                    let friction = mu * G * screen.config.slope_deg.to_radians().cos()
                        + self.transport_adhesion(&p, cell.depth_m);
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

#[cfg(test)]
mod transport_tests {
    use super::*;
    fn parcel(volume: f64) -> ScreenParticle {
        let dry = crate::Material {
            soil_m3: volume,
            ..Default::default()
        };
        let material = crate::Material {
            pore_water_m3: dry.pore_volume_m3() * 0.7,
            ..dry
        };
        ScreenParticle {
            id: 1,
            grain: crate::sluice::Grain {
                kind: crate::GrainKind::Sand,
                material,
                diameter_m: 0.00002,
                position: Default::default(),
                velocity: Default::default(),
                affine: Default::default(),
                rotation: Default::default(),
                angular_velocity: Default::default(),
                resting: false,
                volume_resolved: true,
            },
            position: DVec3::ZERO,
            velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
            in_gutter: false,
            passed_deck: false,
        }
    }
    #[test]
    fn thick_wet_sheet_basal_traction_is_area_limited_and_carries_water_inertia() {
        let state = BulkState::new(BulkDeck::default());
        let area = 0.1;
        let mut previous = 0.;
        for depth in [0.002, 0.02, 0.2, 2.] {
            let p = parcel(area * depth);
            let m = p.grain.material;
            let force = state.transport_adhesion(&p, depth) * m.wet_mass_kg();
            let area_bound =
                1.5 * state.config.soil_yield_pa * m.saturation() * m.solid_volume_m3() / depth;
            let expected = area_bound * depth / (depth + p.grain.diameter_m as f64);
            assert!((force - expected).abs() < 1e-12);
            assert!(force <= area_bound && force > previous);
            previous = force;
        }
        let thin = parcel(area * 0.02);
        let deep = parcel(area * 2.);
        let a = state.transport_adhesion(&thin, 0.02) * thin.grain.material.wet_mass_kg();
        let b = state.transport_adhesion(&deep, 2.) * deep.grain.material.wet_mass_kg();
        assert!(
            b / a < 1.002,
            "a hundred layers must not give a hundred times the basal adhesion"
        );
        let old_style = state.adhesion(&deep) * deep.grain.material.wet_mass_kg()
            / (state.adhesion(&thin) * thin.grain.material.wet_mass_kg());
        assert!(
            (old_style - 100.).abs() < 1e-10,
            "negative control exposes all-layers-as-ground error"
        );
    }
    #[test]
    fn legacy_traction_and_isolated_new_grain_limits_are_retained() {
        let p = parcel(0.0001);
        let modern = BulkState::new(BulkDeck::default());
        assert_eq!(modern.transport_adhesion(&p, 0.), modern.adhesion(&p));
        let old = BulkState::new(BulkDeck {
            projection: BulkProjection::CellBinsV1,
            ..Default::default()
        });
        assert_eq!(old.transport_adhesion(&p, 1.), old.adhesion(&p));
    }
}
