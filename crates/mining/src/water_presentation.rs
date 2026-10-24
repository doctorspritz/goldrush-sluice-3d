//! Bounded render history. The playhead is continuous across packet arrivals;
//! interpolation never changes an authoritative particle or solver clock.
use crate::water::InstanceData;
use bevy::prelude::*;
use std::{collections::VecDeque, sync::Arc};

pub(crate) const MAX_HISTORY: usize = 6;
#[derive(Clone)]
struct Frame {
    serial: u64,
    sim: f64,
    wall: f64,
    particles: Arc<Vec<InstanceData>>,
}
#[derive(Clone, Default)]
pub(crate) struct Presentation {
    frames: VecDeque<Frame>,
    serial: u64,
    pair: Option<(u64, u64)>,
    wall: Option<f64>,
    pub time: f64,
    step: f64,
    interval: f64,
    pub alpha: f32,
    pub underruns: u64,
    pub paired: usize,
}
impl Presentation {
    pub fn push(&mut self, sim: f64, wall: f64, particles: Arc<Vec<InstanceData>>) {
        if let Some(previous) = self.frames.back() {
            let delta = sim - previous.sim;
            if delta <= 0. {
                return;
            }
            let elapsed = (wall - previous.wall).max(0.001);
            if self.step == 0. {
                self.step = delta;
                self.interval = elapsed;
            } else {
                self.step += (delta - self.step) * 0.15;
                self.interval += (elapsed - self.interval) * 0.15;
            }
        } else {
            self.time = sim;
            self.wall = Some(wall);
        }
        self.serial += 1;
        self.frames.push_back(Frame {
            serial: self.serial,
            sim,
            wall,
            particles,
        });
        while self.frames.len() > MAX_HISTORY {
            self.frames.pop_front();
        }
        self.time = self.time.max(self.frames.front().unwrap().sim);
    }
    /// Returns a new upload only when the interpolation endpoints change.
    pub fn advance(&mut self, wall: f64, paused: bool) -> Option<Arc<Vec<InstanceData>>> {
        let dt = (wall - self.wall.replace(wall).unwrap_or(wall)).max(0.);
        let latest = self.frames.back()?;
        if paused {
            self.time = latest.sim;
        } else if self.step > 0. {
            let rate = (self.step / self.interval.max(0.001)).clamp(0.01, 1.5);
            let delay = (self.step * 1.5).clamp(1. / 60., 0.1);
            // Gently recover the buffer after variable packet timing. Never
            // rewind, reset at arrival, or extrapolate beyond known physics.
            let error = latest.sim - delay - self.time;
            let speed = (rate + error * 6.).clamp(0., rate * 1.25);
            let next = self.time + dt * speed;
            if next > latest.sim {
                self.underruns += 1;
            }
            self.time = next.min(latest.sim);
        }
        while self.frames.len() > 2 && self.frames[1].sim <= self.time {
            self.frames.pop_front();
        }
        let previous = self.frames.front()?;
        let current = self.frames.get(1).unwrap_or(previous);
        self.alpha = if current.sim > previous.sim {
            ((self.time - previous.sim) / (current.sim - previous.sim)).clamp(0., 1.) as f32
        } else {
            1.
        };
        let pair = (previous.serial, current.serial);
        if self.pair == Some(pair) {
            return None;
        }
        self.pair = Some(pair);
        self.paired = current.particles.len();
        Some(current.particles.clone())
    }
}

fn vector(a: [f32; 4]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}
fn aligned(previous: &InstanceData, mut current: InstanceData) -> InstanceData {
    let a = [previous.axis_x, previous.axis_y, previous.axis_z];
    let b = [current.axis_x, current.axis_y, current.axis_z];
    let a_unit = a.map(|v| vector(v).normalize_or_zero());
    let b_unit = b.map(|v| vector(v).normalize_or_zero());
    // Each pair contributes to several permutations. Compute the nine dot
    // products once, preserving the score order and tie-breaking exactly.
    let similarity: [[f32; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| a_unit[i].dot(b_unit[j]).abs()));
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let score = |order: &[usize; 3]| (0..3).map(|i| similarity[i][order[i]]).sum::<f32>();
    let order = permutations
        .into_iter()
        .max_by(|a, b| score(a).total_cmp(&score(b)))
        .unwrap();
    let old_normal = Vec3::new(a[0][3], a[1][3], a[2][3]);
    let normal = Vec3::new(b[0][3], b[1][3], b[2][3]);
    let normals = (normal * if normal.dot(old_normal) < 0. { -1. } else { 1. }).to_array();
    for (i, target) in [
        &mut current.axis_x,
        &mut current.axis_y,
        &mut current.axis_z,
    ]
    .into_iter()
    .enumerate()
    {
        let source = vector(b[order[i]]);
        let v = source
            * if vector(a[i]).dot(source) < 0. {
                -1.
            } else {
                1.
            };
        *target = [v.x, v.y, v.z, normals[i]];
    }
    current
}
fn bridge(previous: &InstanceData, current: InstanceData) -> InstanceData {
    let mut p = aligned(previous, current);
    p.previous_center = previous.center_radius;
    p.previous_x = previous.axis_x;
    p.previous_y = previous.axis_y;
    p.previous_z = previous.axis_z;
    p.previous_shape = previous.shape;
    // Terrain uses these slots for slopes; fluid uses them for prior velocity.
    p.slope_x = previous.velocity;
    p
}
/// Both inputs are sorted by persistent identity. Births/deaths grow/shrink
/// their render kernel over the interval; outlet transfers retain their ID.
pub(crate) fn pair_particles(
    previous: &[(u64, InstanceData)],
    current: &[(u64, InstanceData)],
) -> Vec<InstanceData> {
    let mut output = Vec::with_capacity(previous.len() + current.len());
    let (mut a, mut b) = (0, 0);
    while a < previous.len() || b < current.len() {
        let old = previous.get(a);
        let new = current.get(b);
        if let (Some((id_a, p)), Some((id_b, q))) = (old, new) {
            if id_a == id_b {
                output.push(bridge(p, *q));
                a += 1;
                b += 1;
                continue;
            }
        }
        match (old, new) {
            (Some((id, p)), new) if new.is_none_or(|(next, _)| id < next) => {
                let mut q = *p;
                q.center_radius[3] = 0.;
                output.push(bridge(p, q));
                a += 1;
            }
            (_, Some((_, q))) => {
                let mut p = *q;
                p.center_radius[3] = 0.;
                output.push(bridge(&p, *q));
                b += 1;
            }
            _ => unreachable!("nonempty sorted particle inputs"),
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    fn particle(x: f32) -> InstanceData {
        InstanceData {
            center_radius: [x, 0., 0., 0.01],
            axis_x: [1., 0., 0., 0.],
            axis_y: [0., 1., 0., 1.],
            axis_z: [0., 0., 1., 0.],
            ..Default::default()
        }
    }
    #[test]
    fn delayed_and_batched_packets_never_rewind_or_disable_interpolation() {
        for schedule in [
            [0., 0.04, 0.08, 0.16, 0.20, 0.24],
            [0., 0.03, 0.09, 0.14, 0.21, 0.26],
        ] {
            let mut p = Presentation::default();
            let mut sample = 0;
            let mut previous = 0.;
            let mut between = 0;
            for frame in 0..=80 {
                let wall = frame as f64 / 240.;
                while sample < schedule.len() && schedule[sample] <= wall {
                    let sim = sample as f64 / 30.;
                    p.push(sim, wall, Arc::new(vec![particle(sim as f32)]));
                    sample += 1;
                }
                p.advance(wall, false);
                assert!(
                    p.time >= previous && p.time <= (sample.saturating_sub(1)) as f64 / 30. + 1e-9
                );
                if p.alpha > 0. && p.alpha < 1. {
                    between += 1;
                }
                previous = p.time;
            }
            assert!(
                between > 30,
                "continuous frames between irregular packets: {between}"
            );
            p.advance(1., true);
            assert_eq!(p.alpha, 1.);
        }
    }
    #[test]
    fn correspondence_survives_removal_birth_and_channel_to_outlet_transfer() {
        let a = [(10, particle(0.)), (20, particle(1.))];
        let mut outlet = particle(2.);
        outlet.shape[2] = 2.;
        let b = [(20, outlet), (30, particle(3.))];
        let pairs = pair_particles(&a, &b);
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[0].center_radius[3], 0.);
        assert_eq!(pairs[1].previous_center[0], 1.);
        assert_eq!(pairs[1].center_radius[0], 2.);
        assert_eq!(pairs[2].previous_center[3], 0.);
    }
    #[test]
    fn equivalent_eigenvector_signs_and_order_do_not_collapse_the_surface() {
        let p = particle(0.);
        let mut q = p;
        q.axis_x = [0., -1., 0., 0.];
        q.axis_y = [0., 0., -1., 1.];
        q.axis_z = [-1., 0., 0., 0.];
        let q = bridge(&p, q);
        for (a, b) in [
            (p.axis_x, q.axis_x),
            (p.axis_y, q.axis_y),
            (p.axis_z, q.axis_z),
        ] {
            assert!(vector(a).dot(vector(b)) > 0.99);
        }
    }
}
