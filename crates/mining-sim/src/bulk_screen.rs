//! Statistical bulk-deck approximation, deliberately separate from wire DEM.
//! Finite weighted parcels own all stock. The coarse bed only derives depth and
//! adhesive coverage; it never creates inventory. Passage is a saved deterministic
//! residence hazard, not a recovery bonus. Parameters remain uncalibrated.
use super::{rotate_grain, BulkBedField, BulkProjection, Screen, ScreenParticle, G};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BulkContact {
    /// Saved models 2/3: point friction removes translation without rolling.
    #[default]
    PointFrictionV1,
    /// Saved model 4: spherical basal contact couples translation and spin.
    RollingSpheresV2,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BulkDeck {
    pub cells_downstream: u16,
    pub cells_across: u16,
    /// Assumed yield stress of saturated soil fines; not a measured clay law.
    pub soil_yield_pa: f64,
    /// Missing in saved model 2; never silently migrate its force law.
    #[serde(default)]
    pub projection: BulkProjection,
    /// Missing in older saves; retain their point-friction law exactly.
    #[serde(default)]
    pub contact: BulkContact,
}
impl Default for BulkDeck {
    fn default() -> Self {
        Self {
            cells_downstream: 16,
            cells_across: 4,
            soil_yield_pa: 25.,
            projection: BulkProjection::SmoothCellsV2,
            contact: BulkContact::RollingSpheresV2,
        }
    }
}
impl BulkDeck {
    pub(super) fn valid(&self) -> bool {
        (1..=64).contains(&self.cells_downstream)
            && (1..=16).contains(&self.cells_across)
            && (0.0..=200.).contains(&self.soil_yield_pa)
            && !(self.contact == BulkContact::RollingSpheresV2
                && self.projection == BulkProjection::CellBinsV1)
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
                    let incident_normal_speed = (-relative.dot(normal)).max(0.);
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
                    // A new landing already advanced through this interval's
                    // gravity and world translation. Do not integrate it twice.
                    let landed_rolling =
                        landed && self.config.contact == BulkContact::RollingSpheresV2;
                    if !landed_rolling {
                        relative += acceleration * h;
                    }
                    let friction = mu * G * screen.config.slope_deg.to_radians().cos()
                        + self.transport_adhesion(&p, cell.depth_m);
                    let speed = relative.length();
                    if self.config.contact == BulkContact::RollingSpheresV2 {
                        let drag = self.transport_adhesion(&p, cell.depth_m);
                        if adhesive <= drive {
                            // The same impulse acts on the centre and on its
                            // radius arm. Static contact can roll, not freeze.
                            rolling_friction(
                                &mut relative,
                                &mut p.angular_velocity,
                                normal,
                                r,
                                mu * if landed_rolling {
                                    incident_normal_speed
                                } else {
                                    G * screen.config.slope_deg.to_radians().cos() * h
                                },
                            );
                        }
                        // Bonded fines retain the declared cohesive arrest;
                        // unbonded wet contact uses the existing adhesion drag.
                        // This is not a calibrated rolling-resistance law.
                        let resistance = if adhesive > drive { friction } else { drag };
                        if resistance > 0. {
                            let factor =
                                (1. - resistance * h / relative.length().max(1e-30)).max(0.);
                            relative *= factor;
                            let torsion = normal * p.angular_velocity.dot(normal);
                            p.angular_velocity = torsion + (p.angular_velocity - torsion) * factor;
                        }
                    } else {
                        relative *= (1. - friction * h / speed.max(1e-30)).max(0.);
                    }
                    if !landed_rolling {
                        p.position += (off1 - off0) + relative * h;
                    }
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
                p.grain.resting = residence.on_deck
                    && !screen.motor_on
                    && (self.config.contact == BulkContact::PointFrictionV1
                        || (p.velocity.length_squared() < 1e-12
                            && p.angular_velocity.length_squared() < 1e-12));
                if self.config.contact == BulkContact::RollingSpheresV2 {
                    p.grain.angular_velocity = p.angular_velocity.as_vec3();
                }
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
/// Coulomb impulse per unit mass for a spherical contact, I/m = 2r²/5.
/// The contact mobility is 1 + mr²/I = 7/2. Work is dissipative in the
/// deck frame; a moving deck may supply the corresponding impulse work.
fn rolling_friction(
    velocity: &mut DVec3,
    spin: &mut DVec3,
    normal: DVec3,
    radius: f64,
    maximum_delta_v: f64,
) {
    let slip = *velocity + spin.cross(-normal * radius);
    let impulse = -slip * (1. / 3.5_f64).min(maximum_delta_v / slip.length().max(1e-30));
    *velocity += impulse;
    *spin += -normal.cross(impulse) * (2.5 / radius);
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
            contact: BulkContact::PointFrictionV1,
            ..Default::default()
        });
        assert_eq!(old.transport_adhesion(&p, 1.), old.adhesion(&p));
    }
}

#[cfg(test)]
mod rolling_tests {
    use super::*;
    #[test]
    fn no_slip_plane_has_five_sevenths_downhill_acceleration() {
        for angle in [1.0_f64, 12., 25.] {
            let theta = angle.to_radians();
            let normal = DVec3::new(theta.sin(), theta.cos(), 0.);
            let downhill = DVec3::new(theta.cos(), -theta.sin(), 0.);
            for radius in [0.00005, 0.003, 0.01] {
                let mut v = DVec3::ZERO;
                let mut spin = DVec3::ZERO;
                for _ in 0..240 {
                    v += downhill * (G * theta.sin() / 240.);
                    rolling_friction(&mut v, &mut spin, normal, radius, G / 240.);
                }
                let expected = downhill * (5. / 7. * G * theta.sin());
                assert!(v.distance(expected) < 1e-11);
                assert!((v + spin.cross(-normal * radius)).length() < 1e-11);
                assert!(spin.distance(normal.cross(expected) / radius) < 1e-7);
            }
        }
    }
    #[test]
    fn contact_impulse_obeys_coulomb_bound_pivot_momentum_and_relative_energy() {
        for i in 0..128 {
            let normal = DVec3::new(0.2 + (i as f64).sin(), 1., (i as f64 * 0.7).cos()).normalize();
            let radius = 0.00005 + i as f64 * 0.0001;
            let seed = DVec3::new((i as f64 * 0.3).cos(), 0.2, (i as f64 * 0.9).sin());
            let v0 = seed - normal * seed.dot(normal);
            let w0 = DVec3::new(23., -71., i as f64 - 60.);
            for cap in [0., 0.00001, 0.1, 10.] {
                let mut v = v0;
                let mut spin = w0;
                let inertia = 0.4 * radius * radius;
                let energy0 = 0.5 * (v0.length_squared() + inertia * w0.length_squared());
                let momentum0 = w0 * inertia + (normal * radius).cross(v0);
                rolling_friction(&mut v, &mut spin, normal, radius, cap);
                let energy1 = 0.5 * (v.length_squared() + inertia * spin.length_squared());
                let momentum1 = spin * inertia + (normal * radius).cross(v);
                assert!(energy1 <= energy0 + 2e-14);
                assert!((v - v0).length() <= cap + 1e-14);
                assert!(momentum1.distance(momentum0) < 2e-15);
                assert!((v - v0).dot(normal).abs() < 1e-14);
            }
        }
    }
    #[test]
    fn existing_pure_roll_and_normal_spin_are_not_damped_as_sliding() {
        let radius = 0.002;
        let normal = DVec3::Y;
        let mut spin = DVec3::new(0., 12., -500.);
        let mut v = DVec3::X;
        let initial = (v, spin);
        rolling_friction(&mut v, &mut spin, normal, radius, 1.);
        assert_eq!((v, spin), initial);
    }
}
