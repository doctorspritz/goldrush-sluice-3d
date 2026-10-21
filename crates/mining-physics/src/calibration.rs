//! Repeatable physical experiments. Residence, actual cleanup and outlet loss
//! are separate measurements; no expected recovery is injected into the solver.
use mining_sim::{
    sluice::{ContactModel, GrainKind},
    Session,
};
use serde_json::{json, Value};

pub const DIAMETERS_M: [f32; 4] = [0.00008, 0.00015, 0.0003, 0.0005];
pub const TRACERS_PER_SIZE: usize = 24;

#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub name: &'static str,
    pub flow_m3s: f32,
    pub feed_m3s: f32,
    pub parcel_m3: f64,
    pub riffles: bool,
    pub contact: ContactModel,
}
pub const CASES: [Case; 7] = [
    Case {
        name: "low",
        flow_m3s: 0.00125,
        feed_m3s: 0.0001,
        parcel_m3: 2e-6,
        riffles: true,
        contact: ContactModel::Coulomb,
    },
    Case {
        name: "nominal",
        flow_m3s: 0.00375,
        feed_m3s: 0.0001,
        parcel_m3: 2e-6,
        riffles: true,
        contact: ContactModel::Coulomb,
    },
    Case {
        name: "high",
        flow_m3s: 0.0075,
        feed_m3s: 0.0001,
        parcel_m3: 2e-6,
        riffles: true,
        contact: ContactModel::Coulomb,
    },
    Case {
        name: "heavy_feed",
        flow_m3s: 0.00375,
        feed_m3s: 0.0004,
        parcel_m3: 2e-6,
        riffles: true,
        contact: ContactModel::Coulomb,
    },
    Case {
        name: "bare",
        flow_m3s: 0.00375,
        feed_m3s: 0.0001,
        parcel_m3: 2e-6,
        riffles: false,
        contact: ContactModel::Coulomb,
    },
    Case {
        name: "parcel_coarse",
        flow_m3s: 0.00375,
        feed_m3s: 0.00002,
        parcel_m3: 2e-6,
        riffles: true,
        contact: ContactModel::Coulomb,
    },
    Case {
        name: "parcel_fine",
        flow_m3s: 0.00375,
        feed_m3s: 0.00002,
        parcel_m3: 0.5e-6,
        riffles: true,
        contact: ContactModel::Coulomb,
    },
];

#[derive(Clone, Debug, Default)]
pub struct Assay {
    pub gold_g: [f64; 4],
    pub quartz_m3: [f64; 4],
    pub count: [usize; 4],
}
impl Assay {
    pub fn resident(s: &Session) -> Self {
        let mut assay = Self::default();
        for g in &s.sluice.grains {
            let Some(bin) = DIAMETERS_M
                .iter()
                .position(|d| (*d - g.diameter_m).abs() < 1e-9)
            else {
                continue;
            };
            match g.kind {
                GrainKind::Gold => {
                    assay.gold_g[bin] += g.material.gold_g;
                    assay.count[bin] += 1;
                }
                GrainKind::Sand => {
                    assay.quartz_m3[bin] += g.material.sand_m3;
                }
                GrainKind::Gravel => {}
            }
        }
        assay
    }
    pub fn json(&self) -> Value {
        json!({"diameters_m":DIAMETERS_M,"gold_g":self.gold_g,"quartz_m3":self.quartz_m3,"resident_gold_count":self.count})
    }
}

/// Spatial samples of the lee pockets; these are backflow indicators, not
/// a claim of grid-converged turbulence or a complete vorticity measurement.
pub fn observe(s: &Session) -> Value {
    let mut count = 0;
    let mut reverse = 0;
    let mut slow = 0;
    let mut bed = [0.; 4];
    let mut peak = [0.0_f32; 4];
    for p in &s.sluice.water {
        let x = p.position.x;
        let local = s.sluice.grid_to_local(p.position);
        if (0.42..1.35).contains(&x) && ((x - 0.4) % 0.32) < 0.12 && local.y < 0.035 {
            count += 1;
            reverse += usize::from(p.velocity.x < -0.02);
            slow += usize::from(p.velocity.length() < 0.15);
        }
    }
    for g in &s.sluice.grains {
        let local = s.sluice.grid_to_local(g.position);
        if (0.4..1.5).contains(&g.position.x) && g.resting {
            let i = ((g.position.x - 0.4) / 0.32) as usize;
            bed[i] += g.solid_volume();
            peak[i] = peak[i].max(local.y + g.collision_radius());
        }
    }
    json!({"sim_seconds":s.sim_time,"pocket_samples":count,
        "pocket_reverse_fraction":reverse as f64/count.max(1) as f64,
        "pocket_slow_fraction":slow as f64/count.max(1) as f64,
        "settled_volume_by_riffle_m3":bed,"settled_peak_by_riffle_m":peak,
        "water_downstream_mps":s.sluice.water_speed(),"grains":s.sluice.grains.len(),
        "processed_m3":s.sluice.processed_m3,
        "water_pumped_m3":s.sluice.water_used_m3,
        "whole_water_error_m3":s.water_balance_error_m3(),
        "gold_error_g":s.accounted_gold()-s.initial_gold_g})
}
