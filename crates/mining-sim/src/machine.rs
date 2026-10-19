use crate::{Material, Site, Vec3};
use serde::{Deserialize, Serialize};

pub const BUCKET_CAPACITY: f64 = 0.28;
pub const BOOM_LENGTH: f32 = 3.0;
pub const STICK_LENGTH: f32 = 2.3;

#[derive(Clone, Copy, Debug, Default)]
pub struct MachineInput {
    pub drive: f32,
    pub steer: f32,
    pub swing: f32,
    pub boom: f32,
    pub stick: f32,
    pub curl: f32,
    pub cut: bool,
    pub dump: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Excavator {
    pub position: Vec3,
    pub heading: f32,
    pub swing: f32,
    pub boom: f32,
    pub stick: f32,
    pub curl: f32,
    pub bucket: Material,
    pub engine_on: bool,
    pub fuel_l: f32,
    pub distance_m: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ArmPose {
    pub shoulder: Vec3,
    pub elbow: Vec3,
    pub wrist: Vec3,
    pub lip: Vec3,
    pub angle: f32,
    pub forward: Vec3,
}

impl Excavator {
    pub fn new(site: &Site) -> Self {
        Self {
            position: Vec3::new(19.0, site.height(19.0, 17.0), 17.0),
            heading: 0.0,
            swing: 0.0,
            boom: 0.48,
            stick: -1.30,
            curl: -0.20,
            bucket: Material::default(),
            engine_on: true,
            fuel_l: 45.0,
            distance_m: 0.0,
        }
    }
    pub fn forward(&self) -> Vec3 {
        Vec3::new(self.heading.sin(), 0.0, self.heading.cos())
    }
    pub fn arm(&self) -> ArmPose {
        let yaw = self.heading + self.swing;
        let f = Vec3::new(yaw.sin(), 0.0, yaw.cos());
        let shoulder = self.position + Vec3::Y * 1.65 + f * 0.35;
        let segment =
            |angle: f32, len: f32| f * (angle.cos() * len) + Vec3::Y * (angle.sin() * len);
        let elbow = shoulder + segment(self.boom, BOOM_LENGTH);
        let wrist = elbow + segment(self.boom + self.stick, STICK_LENGTH);
        let angle = self.boom + self.stick + self.curl;
        let lip = wrist + segment(angle, 0.85);
        ArmPose {
            shoulder,
            elbow,
            wrist,
            lip,
            angle,
            forward: f,
        }
    }
    pub fn step(
        &mut self,
        input: MachineInput,
        site: &mut Site,
        dt: f32,
    ) -> Option<(Vec3, Vec3, Material)> {
        if !self.engine_on || self.fuel_l <= 0.0 {
            return None;
        }
        let activity = input.drive.abs()
            + input.steer.abs()
            + input.swing.abs()
            + input.boom.abs()
            + input.stick.abs()
            + input.curl.abs();
        self.fuel_l = (self.fuel_l - dt * (0.0003 + activity.min(3.0) * 0.001)).max(0.0);
        self.heading = (self.heading + input.steer * dt * 0.45).rem_euclid(std::f32::consts::TAU);
        let next = self.position + self.forward() * input.drive * dt * 1.6;
        if Site::bounds(next, 2.0) && site.water_depth(next.x, next.z) < 0.35 {
            let heights = [(-0.85, -1.15), (0.85, -1.15), (-0.85, 1.15), (0.85, 1.15)]
                .map(|(x, z)| site.height(next.x + x, next.z + z));
            let min = heights.iter().copied().fold(f32::INFINITY, f32::min);
            let max = heights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max - min < 0.72 && (max - self.position.y).abs() < 0.70 {
                self.distance_m += (next - self.position).length();
                self.position = Vec3::new(next.x, max, next.z);
            }
        }
        self.swing = (self.swing + input.swing * dt * 0.65).rem_euclid(std::f32::consts::TAU);
        self.boom = (self.boom + input.boom * dt * 0.42).clamp(-0.24, 1.20);
        self.stick = (self.stick + input.stick * dt * 0.55).clamp(-2.15, -0.15);
        self.curl = (self.curl + input.curl * dt * 0.90).clamp(-1.70, 1.45);
        let arm = self.arm();
        let ground = site.height(arm.lip.x, arm.lip.z);
        if (input.cut || input.curl > 0.0) && arm.lip.y < ground + 0.12 && arm.lip.y > ground - 0.65
        {
            let batch = site.dig(
                arm.lip,
                0.44,
                dt * 0.46,
                BUCKET_CAPACITY - self.bucket.volume(),
            );
            self.bucket.add(batch);
        }
        if input.dump && !self.bucket.is_empty() {
            // Opening the bucket releases its real contents at the bucket mouth.
            self.curl = (self.curl - dt * 1.7).max(-1.7);
            return Some((
                arm.wrist + arm.forward * 0.35 - Vec3::Y * 0.10,
                arm.forward * 0.35,
                self.bucket.take_volume((dt * 0.32) as f64),
            ));
        }
        None
    }
    pub fn valid(&self) -> bool {
        Site::bounds(self.position, 0.0)
            && self.position.y.abs() < 100.0
            && [
                self.heading,
                self.swing,
                self.boom,
                self.stick,
                self.curl,
                self.fuel_l,
                self.distance_m,
            ]
            .iter()
            .all(|v| v.is_finite())
            && (-0.24..=1.20).contains(&self.boom)
            && (-2.15..=-0.15).contains(&self.stick)
            && (-1.70..=1.45).contains(&self.curl)
            && (0.0..=45.0).contains(&self.fuel_l)
            && self.bucket.valid()
            && self.bucket.volume() <= BUCKET_CAPACITY + 1e-5
    }
}
