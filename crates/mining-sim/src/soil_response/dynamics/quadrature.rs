//! Positive quadrature of endpoint-gap pressure and discrete material work.
//! Splitting at g0/g1, their difference and shifted pressure endpoints makes
//! elastic work and capped viscous loss piecewise quadratic. Pressure itself
//! can be rational during entry/exit, so force/moment quadrature is adaptive.
use super::*;
use std::{cmp::Ordering, collections::BinaryHeap};
#[derive(Default, Clone, Copy)]
struct Point {
    local: DVec3,
    g0: f64,
    g1: f64,
}
impl Point {
    fn lerp(self, b: Self, t: f64) -> Self {
        Self {
            local: (1. - t) * self.local + t * b.local,
            g0: (1. - t) * self.g0 + t * b.g0,
            g1: (1. - t) * self.g1 + t * b.g1,
        }
    }
}
#[derive(Clone, Copy, Default)]
struct Polygon {
    points: [Point; 8],
    len: usize,
}
impl Polygon {
    fn push(&mut self, p: Point) -> Result<(), String> {
        if self.len > 0 && self.points[self.len - 1].local == p.local {
            return Ok(());
        }
        if self.len >= 8 {
            return Err("Endpoint partition vertex bound exceeded".into());
        }
        self.points[self.len] = p;
        self.len += 1;
        Ok(())
    }
    fn clip(self, f: impl Fn(Point) -> f64) -> Result<Self, String> {
        let mut out = Self::default();
        if self.len == 0 {
            return Ok(out);
        }
        let mut a = self.points[self.len - 1];
        let mut sa = f(a);
        for &b in &self.points[..self.len] {
            let sb = f(b);
            if (sa <= 0.) != (sb <= 0.) {
                let scale = sa.abs().max(sb.abs());
                out.push(a.lerp(b, (sa / scale) / (sa / scale - sb / scale)))?;
            }
            if sb <= 0. {
                out.push(b)?;
            }
            a = b;
            sa = sb;
        }
        if out.len > 1 && out.points[0].local == out.points[out.len - 1].local {
            out.len -= 1;
        }
        Ok(out)
    }
}
#[derive(Default, Clone, Copy)]
struct Integral {
    force: f64,
    moment: DVec3,
    loss: f64,
    elastic_work: f64,
    normal_work: f64,
}
impl Integral {
    fn add(&mut self, b: Self) {
        self.force += b.force;
        self.moment += b.moment;
        self.loss += b.loss;
        self.elastic_work += b.elastic_work;
        self.normal_work += b.normal_work;
    }
}
#[derive(Default)]
pub(super) struct Contact {
    pub forces: [DVec3; 2],
    pub torques: [DVec3; 2],
    pub loss: f64,
    pub elastic_work: f64,
    pub normal_work: f64,
    pub estimated_force_moment_error_n: f64,
    pub samples: usize,
    pub depth: u32,
}
struct Rule {
    k: f64,
    c: f64,
    h: f64,
    length: f64,
    settings: Settings,
    samples: usize,
    depth: u32,
}
fn elastic_pressure(k: f64, g0: f64, g1: f64) -> f64 {
    let a = (-g0).max(0.);
    let b = (-g1).max(0.);
    if g0 <= 0. && g1 <= 0. {
        0.5 * k * (a + b)
    } else if g0 >= 0. && g1 >= 0. {
        0.
    } else {
        let compression = a + b;
        0.5 * k * compression * (compression / (g1 - g0).abs())
    }
}
fn contact_fraction(g0: f64, g1: f64) -> f64 {
    if g0 <= 0. && g1 <= 0. {
        1.
    } else if g0 >= 0. && g1 >= 0. {
        0.
    } else {
        ((-g0).max(0.) + (-g1).max(0.)) / (g1 - g0).abs()
    }
}
// Positive symmetric degree-five rule, Dunavant (1985), p.1140.
// https://doi.org/10.1002/nme.1620210612
// Algebraic coefficients avoid limiting exact polynomial work to a shortened
// decimal table. Every node remains inside the material triangle.
fn nodes() -> [([f64; 3], f64); 7] {
    let root = 15f64.sqrt();
    let a = (6. + root) / 21.;
    let b = 1. - 2. * a;
    let w = (155. + root) / 1200.;
    let c = (6. - root) / 21.;
    let d = 1. - 2. * c;
    let v = (155. - root) / 1200.;
    [
        ([1. / 3.; 3], 9. / 40.),
        ([b, a, a], w),
        ([a, b, a], w),
        ([a, a, b], w),
        ([d, c, c], v),
        ([c, d, c], v),
        ([c, c, d], v),
    ]
}
impl Rule {
    fn sample(&self, p: Point) -> Result<Integral, String> {
        let dg = p.g1 - p.g0;
        let elastic = elastic_pressure(self.k, p.g0, p.g1);
        let pressure = if dg <= 0. {
            elastic - self.c * (dg / self.h) * contact_fraction(p.g0, p.g1)
        } else {
            let shift = self.c / (self.k * self.h) * dg;
            elastic_pressure(self.k, p.g0 + shift, p.g1 + shift)
        };
        let result = Integral {
            force: pressure,
            moment: p.local * pressure,
            loss: (elastic - pressure) * dg,
            elastic_work: elastic * dg,
            normal_work: pressure * dg,
        };
        if ![
            pressure,
            result.loss,
            result.elastic_work,
            result.normal_work,
        ]
        .iter()
        .all(|v| v.is_finite())
            || !result.moment.is_finite()
        {
            return Err("Endpoint pressure overflow".into());
        }
        Ok(result)
    }
    fn rule(&mut self, t: [Point; 3]) -> Result<Integral, String> {
        self.samples += 7;
        if self.samples > 300_000 {
            return Err("Patch quadrature work bound exceeded".into());
        }
        let area = 0.5
            * (t[1].local - t[0].local)
                .cross(t[2].local - t[0].local)
                .length();
        let mut result = Integral::default();
        for (coordinates, weight) in nodes() {
            let [x, y, z] = coordinates;
            let [a, b, c] = t;
            let point = Point {
                local: x * a.local + y * b.local + z * c.local,
                g0: x * a.g0 + y * b.g0 + z * c.g0,
                g1: x * a.g1 + y * b.g1 + z * c.g1,
            };
            let sample = self.sample(point)?;
            let weight = area * weight;
            result.force += weight * sample.force;
            result.moment += weight * sample.moment;
            result.loss += weight * sample.loss;
            result.elastic_work += weight * sample.elastic_work;
            result.normal_work += weight * sample.normal_work;
        }
        if ![
            result.force,
            result.loss,
            result.elastic_work,
            result.normal_work,
        ]
        .iter()
        .all(|v| v.is_finite())
            || !result.moment.is_finite()
        {
            return Err("Patch integral overflow".into());
        }
        Ok(result)
    }
    fn panel(
        &mut self,
        t: [Point; 3],
        coarse: Integral,
        depth: u32,
        serial: usize,
    ) -> Result<Panel, String> {
        self.depth = self.depth.max(depth);
        let children = children(t);
        let mut values = [Integral::default(); 4];
        let mut fine = Integral::default();
        for i in 0..4 {
            values[i] = self.rule(children[i])?;
            fine.add(values[i]);
        }
        let error = (fine.force - coarse.force).abs()
            + (fine.moment - coarse.moment).length() / self.length;
        Ok(Panel {
            triangle: t,
            values,
            fine,
            error,
            depth,
            serial,
        })
    }
}
fn children([a, b, c]: [Point; 3]) -> [[Point; 3]; 4] {
    let ab = a.lerp(b, 0.5);
    let bc = b.lerp(c, 0.5);
    let ca = c.lerp(a, 0.5);
    [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
}
// Refine the largest estimated force/moment error in the whole patch. Assigning
// each triangle a proportional area budget spends unattainable relative
// accuracy near a common zero-gap vertex, even when its absolute force is tiny.
struct Panel {
    triangle: [Point; 3],
    values: [Integral; 4],
    fine: Integral,
    error: f64,
    depth: u32,
    serial: usize,
}
impl PartialEq for Panel {
    fn eq(&self, b: &Self) -> bool {
        self.serial == b.serial
    }
}
impl Eq for Panel {}
impl PartialOrd for Panel {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
impl Ord for Panel {
    fn cmp(&self, b: &Self) -> Ordering {
        self.error
            .total_cmp(&b.error)
            .then_with(|| b.serial.cmp(&self.serial))
    }
}
struct Budget {
    queue: BinaryHeap<Panel>,
    integral: Integral,
    error: f64,
    serial: usize,
}
impl Budget {
    fn new() -> Self {
        Self {
            queue: BinaryHeap::new(),
            integral: Integral::default(),
            error: 0.,
            serial: 0,
        }
    }
    fn add(
        &mut self,
        rule: &mut Rule,
        t: [Point; 3],
        coarse: Integral,
        depth: u32,
    ) -> Result<(), String> {
        let panel = rule.panel(t, coarse, depth, self.serial)?;
        self.serial += 1;
        self.integral.force += panel.fine.force;
        self.integral.moment += panel.fine.moment;
        // Work and capped loss are quadratic on each admitted partition.
        // Retain their exact root rule, rather than repeatedly subtracting
        // and re-adding the same integral during force-only refinement.
        if depth == 0 {
            self.integral.loss += panel.fine.loss;
            self.integral.elastic_work += panel.fine.elastic_work;
            self.integral.normal_work += panel.fine.normal_work;
        }
        self.error += panel.error;
        if depth < rule.settings.maximum_quadrature_depth {
            self.queue.push(panel);
        }
        Ok(())
    }
    fn finish(mut self, rule: &mut Rule) -> Result<(Integral, f64), String> {
        loop {
            let tolerance = rule.settings.quadrature_absolute_force_tolerance_n
                + rule.settings.quadrature_relative_tolerance * self.integral.force.abs();
            if self.error <= tolerance {
                return Ok((self.integral, self.error));
            }
            let panel = self.queue.pop().ok_or_else(|| {
                format!(
                    "Global patch force quadrature did not converge: {} > {tolerance}",
                    self.error
                )
            })?;
            self.integral.force -= panel.fine.force;
            self.integral.moment -= panel.fine.moment;
            self.error -= panel.error;
            let children = children(panel.triangle);
            for (child, value) in children.into_iter().zip(panel.values) {
                self.add(rule, child, value, panel.depth + 1)?;
            }
        }
    }
}

pub(super) fn integrate(
    layer: NormalLayer,
    patch: &SurfacePatch,
    plane: Plane,
    before: [RigidMotion; 2],
    after: [RigidMotion; 2],
    h: f64,
    settings: Settings,
) -> Result<Contact, String> {
    let n0 = before[1].rotation * plane.normal;
    let n1 = after[1].rotation * plane.normal;
    let d0 =
        before[0].center_of_mass_m - before[1].center_of_mass_m - before[1].rotation * plane.point;
    let d1 =
        after[0].center_of_mass_m - after[1].center_of_mass_m - after[1].rotation * plane.point;
    let shift = layer.viscosity_density / (layer.stiffness_density * h);
    if !shift.is_finite() {
        return Err("Endpoint viscosity scale overflow".into());
    }
    let length = patch
        .triangles
        .iter()
        .flat_map(|t| t.iter())
        .map(|p| p.length())
        .fold(0f64, f64::max);
    let mut rule = Rule {
        k: layer.stiffness_density,
        c: layer.viscosity_density,
        h,
        length,
        settings,
        samples: 0,
        depth: 0,
    };
    let mut budget = Budget::new();
    for triangle in &patch.triangles {
        let mut initial = Polygon::default();
        for &local in triangle {
            let g0 = n0.dot(d0 + before[0].rotation * local);
            let g1 = n1.dot(d1 + after[0].rotation * local);
            if !g0.is_finite() || !g1.is_finite() {
                return Err("Nonfinite endpoint gap".into());
            }
            initial.push(Point { local, g0, g1 })?;
        }
        let mut polygons = [Polygon::default(); 32];
        polygons[0] = initial;
        let mut count = 1;
        for field in 0..5 {
            let signed = |p: Point| match field {
                0 => p.g0,
                1 => p.g1,
                2 => p.g1 - p.g0,
                3 => p.g0 + shift * (p.g1 - p.g0),
                _ => p.g1 + shift * (p.g1 - p.g0),
            };
            let mut next = [Polygon::default(); 32];
            let mut next_count = 0;
            for poly in &polygons[..count] {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for &point in &poly.points[..poly.len] {
                    let v = signed(point);
                    if !v.is_finite() {
                        return Err("Endpoint partition overflow".into());
                    }
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                if lo < 0. && hi > 0. {
                    if next_count + 2 > 32 {
                        return Err("Endpoint partition count exceeded".into());
                    }
                    next[next_count] = poly.clip(signed)?;
                    next[next_count + 1] = poly.clip(|p| -signed(p))?;
                    next_count += 2;
                } else {
                    if next_count >= 32 {
                        return Err("Endpoint partition count exceeded".into());
                    }
                    next[next_count] = *poly;
                    next_count += 1;
                }
            }
            polygons = next;
            count = next_count;
        }
        for poly in &polygons[..count] {
            for i in 1..poly.len.saturating_sub(1) {
                let t = [poly.points[0], poly.points[i], poly.points[i + 1]];
                let coarse = rule.rule(t)?;
                budget.add(&mut rule, t, coarse, 0)?;
            }
        }
    }
    let (integral, estimated_error) = budget.finish(&mut rule)?;
    // Do not normalize either endpoint average: these discrete Jacobians obey
    // the exact Cayley endpoint-gap increment and its force/torque work pairing.
    let normal = (n0 + n1) * 0.5;
    let moment = (before[0].rotation * integral.moment + after[0].rotation * integral.moment) * 0.5;
    let delta = (before[0].center_of_mass_m + after[0].center_of_mass_m
        - before[1].center_of_mass_m
        - after[1].center_of_mass_m)
        * 0.5;
    let force = normal * integral.force;
    let torque_a = moment.cross(normal);
    let torque_b = -(delta * integral.force + moment).cross(normal);
    Ok(Contact {
        forces: [force, -force],
        torques: [torque_a, torque_b],
        loss: integral.loss,
        elastic_work: integral.elastic_work,
        normal_work: integral.normal_work,
        estimated_force_moment_error_n: estimated_error,
        samples: rule.samples,
        depth: rule.depth,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rule(k: f64, c: f64, settings: Settings) -> Rule {
        Rule {
            k,
            c,
            h: 1.,
            length: 2f64.sqrt(),
            settings,
            samples: 0,
            depth: 0,
        }
    }
    #[test]
    fn capped_opening_pressure_integrates_the_active_time_instead_of_capping_the_mean() {
        let rule = rule(1000., 500., Settings::default());
        let point = Point {
            local: DVec3::ZERO,
            g0: -0.02,
            g1: -0.001,
        };
        let actual = rule.sample(point).unwrap();
        let velocity = point.g1 - point.g0;
        let independent = (0..100_000)
            .map(|i| {
                let gap = point.g0 + velocity * (i as f64 + 0.5) / 100_000.;
                (-1000. * gap - 500. * velocity).max(0.)
            })
            .sum::<f64>()
            / 100_000.;
        let naive = (10.5 - 500. * velocity).max(0.);
        assert!((actual.force - independent).abs() < 1e-8);
        assert!((actual.force - 2.901315789473684).abs() < 1e-12);
        assert!((actual.loss - 0.144375).abs() < 1e-12);
        assert!((actual.force - naive).abs() > 1.9);
        println!(
            "CAPPED_OPENING pressure={}Pa naive_mean_cap={naive}Pa dissipation={}J/m2",
            actual.force, actual.loss
        );
    }
    #[test]
    fn global_force_budget_resolves_entry_exit_against_an_independent_log_integral() {
        let point = |x, z| Point {
            local: DVec3::new(x, 0., z),
            g0: -0.125 * x,
            g1: 0.0025,
        };
        let triangles = [
            [point(0., 0.), point(1., 0.), point(1., 1.)],
            [point(0., 0.), point(1., 1.), point(0., 1.)],
        ];
        let k = 1e6;
        let t: f64 = 0.0025 / 0.125;
        let logarithm = ((1. + t) / t).ln();
        let force = 0.5 * k * 0.125 * (0.5 - t + t * t * logarithm);
        let moment_x = 0.5 * k * 0.125 * (1. / 3. - t / 2. + t * t - t * t * t * logarithm);
        for relative in [1e-6, 1e-8, 1e-10] {
            let settings = Settings {
                quadrature_relative_tolerance: relative,
                ..Settings::default()
            };
            let mut rule = rule(k, 0., settings);
            let mut budget = Budget::new();
            for triangle in triangles {
                let coarse = rule.rule(triangle).unwrap();
                budget.add(&mut rule, triangle, coarse, 0).unwrap();
            }
            let (actual, estimated) = budget.finish(&mut rule).unwrap();
            let error = (actual.force - force).abs()
                + (actual.moment.x - moment_x).abs()
                + (actual.moment.z - force * 0.5).abs();
            assert!(
                error < 2. * (settings.quadrature_absolute_force_tolerance_n + relative * force),
                "force/moment error={error}N"
            );
            assert!((actual.elastic_work - 0.5 * k * 0.125f64.powi(2) / 3.).abs() < 1e-10);
            assert!(actual.loss.abs() < 1e-10);
            println!("RATIONAL_PATCH relative_target={relative} force_moment_error={error}N estimated={estimated}N samples={} depth={}",rule.samples,rule.depth);
        }
    }
}

#[cfg(test)]
mod clipping_tests {
    use super::*;
    #[test]
    fn shared_zero_gap_vertices_do_not_consume_duplicate_partition_slots() {
        let mut polygon = Polygon::default();
        for local in [DVec3::ZERO, DVec3::new(-1., 0., 0.), DVec3::new(1., 0., 1.)] {
            polygon
                .push(Point {
                    local,
                    ..Default::default()
                })
                .unwrap();
        }
        let clipped = polygon.clip(|p| p.local.x).unwrap();
        assert_eq!(clipped.len, 3, "a zero-gap endpoint was inserted twice");
        let area = 0.5
            * (clipped.points[1].local - clipped.points[0].local)
                .cross(clipped.points[2].local - clipped.points[0].local)
                .length();
        assert!((area - 0.25).abs() < 1e-15);
    }
}

#[cfg(test)]
mod polynomial_tests {
    use super::*;
    #[test]
    fn positive_rule_retains_every_triangle_monomial_through_degree_five() {
        let factorial = |n: u32| (1..=n).map(f64::from).product::<f64>();
        for a in 0..=5 {
            for b in 0..=5 - a {
                let actual = nodes()
                    .into_iter()
                    .map(|([_, x, z], weight)| {
                        assert!(weight > 0. && x >= 0. && z >= 0. && x + z <= 1.);
                        0.5 * weight * x.powi(a as i32) * z.powi(b as i32)
                    })
                    .sum::<f64>();
                let exact = factorial(a) * factorial(b) / factorial(a + b + 2);
                assert!(
                    (actual - exact).abs() < 1e-14,
                    "monomial x^{a} z^{b}: {actual} vs {exact}"
                );
            }
        }
    }
}
