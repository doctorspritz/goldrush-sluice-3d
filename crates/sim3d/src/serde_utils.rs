//! Serde utilities for glam types.

use glam::{Mat3, Quat, Vec3};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Serde proxy for Vec3
#[derive(Serialize, Deserialize)]
pub struct Vec3Def {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl From<Vec3> for Vec3Def {
    fn from(v: Vec3) -> Self {
        Self { x: v.x, y: v.y, z: v.z }
    }
}

impl From<Vec3Def> for Vec3 {
    fn from(def: Vec3Def) -> Self {
        Vec3::new(def.x, def.y, def.z)
    }
}

pub fn serialize_vec3<S>(v: &Vec3, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    Vec3Def::from(*v).serialize(s)
}

pub fn deserialize_vec3<'de, D>(d: D) -> Result<Vec3, D::Error>
where
    D: Deserializer<'de>,
{
    Vec3Def::deserialize(d).map(Vec3::from)
}

/// Serde proxy for Mat3
#[derive(Serialize, Deserialize)]
pub struct Mat3Def {
    pub x_axis: Vec3Def,
    pub y_axis: Vec3Def,
    pub z_axis: Vec3Def,
}

impl From<Mat3> for Mat3Def {
    fn from(m: Mat3) -> Self {
        Self {
            x_axis: m.x_axis.into(),
            y_axis: m.y_axis.into(),
            z_axis: m.z_axis.into(),
        }
    }
}

impl From<Mat3Def> for Mat3 {
    fn from(def: Mat3Def) -> Self {
        Mat3::from_cols(def.x_axis.into(), def.y_axis.into(), def.z_axis.into())
    }
}

pub fn serialize_mat3<S>(m: &Mat3, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    Mat3Def::from(*m).serialize(s)
}

pub fn deserialize_mat3<'de, D>(d: D) -> Result<Mat3, D::Error>
where
    D: Deserializer<'de>,
{
    Mat3Def::deserialize(d).map(Mat3::from)
}

/// Serde proxy for Quat
#[derive(Serialize, Deserialize)]
pub struct QuatDef {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl From<Quat> for QuatDef {
    fn from(q: Quat) -> Self {
        let array = q.to_array();
        Self { x: array[0], y: array[1], z: array[2], w: array[3] }
    }
}

impl From<QuatDef> for Quat {
    fn from(def: QuatDef) -> Self {
        Quat::from_array([def.x, def.y, def.z, def.w])
    }
}

pub fn serialize_quat<S>(q: &Quat, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    QuatDef::from(*q).serialize(s)
}

pub fn deserialize_quat<'de, D>(d: D) -> Result<Quat, D::Error>
where
    D: Deserializer<'de>,
{
    QuatDef::deserialize(d).map(Quat::from)
}
