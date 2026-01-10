//! Test harness for simulation validation
//!
//! Provides a framework for running physics tests with invariant checking.

use std::time::Instant;

/// Result of running a simulation test
#[derive(Debug)]
pub struct TestResult {
    pub passed: bool,
    pub test_name: String,
    pub metrics: TestMetrics,
    pub failures: Vec<String>,
}

/// Metrics collected during a test run
#[derive(Debug, Default)]
pub struct TestMetrics {
    pub frame_count: u32,
    pub particle_count_start: u32,
    pub particle_count_end: u32,
    pub max_velocity: f32,
    pub max_pressure: f32,
    pub nan_detected: bool,
    pub particles_in_solid: u32,
    pub total_divergence: f32,
    pub elapsed_seconds: f32,
}

/// Invariants that can be checked during simulation
#[derive(Debug, Clone)]
pub enum Invariant {
    /// Particle count should stay within tolerance of initial
    ParticleCountStable { tolerance_pct: f32 },
    /// Maximum velocity should stay below limit
    MaxVelocityBelow { limit: f32 },
    /// Maximum pressure should stay below limit
    MaxPressureBelow { limit: f32 },
    /// No NaN values in simulation state
    NoNaN,
    /// No particles inside solid cells
    NoParticlesInSolid,
    /// Total energy should decrease (dissipative system)
    EnergyDecreasing,
    /// Bed height change should stay below limit per frame
    BedHeightChangeBelow { limit: f32 },
    /// Divergence should stay below threshold (incompressibility)
    DivergenceBelow { threshold: f32 },
}

impl Invariant {
    /// Check this invariant against current metrics
    pub fn check(&self, metrics: &TestMetrics, prev_metrics: Option<&TestMetrics>) -> Result<(), String> {
        match self {
            Invariant::ParticleCountStable { tolerance_pct } => {
                if metrics.particle_count_start == 0 {
                    return Ok(());
                }
                let ratio = metrics.particle_count_end as f32 / metrics.particle_count_start as f32;
                if (ratio - 1.0).abs() > *tolerance_pct {
                    return Err(format!(
                        "Particle count unstable: {} -> {} ({:.1}% change)",
                        metrics.particle_count_start,
                        metrics.particle_count_end,
                        (ratio - 1.0) * 100.0
                    ));
                }
            }
            Invariant::MaxVelocityBelow { limit } => {
                if metrics.max_velocity > *limit {
                    return Err(format!(
                        "Max velocity {} exceeds limit {}",
                        metrics.max_velocity, limit
                    ));
                }
            }
            Invariant::MaxPressureBelow { limit } => {
                if metrics.max_pressure > *limit {
                    return Err(format!(
                        "Max pressure {} exceeds limit {}",
                        metrics.max_pressure, limit
                    ));
                }
            }
            Invariant::NoNaN => {
                if metrics.nan_detected {
                    return Err("NaN detected in simulation state".to_string());
                }
            }
            Invariant::NoParticlesInSolid => {
                if metrics.particles_in_solid > 0 {
                    return Err(format!(
                        "{} particles inside solid cells",
                        metrics.particles_in_solid
                    ));
                }
            }
            Invariant::EnergyDecreasing => {
                // TODO: implement energy tracking
            }
            Invariant::BedHeightChangeBelow { limit: _ } => {
                // TODO: implement bed height tracking
            }
            Invariant::DivergenceBelow { threshold } => {
                if metrics.total_divergence > *threshold {
                    return Err(format!(
                        "Divergence {} exceeds threshold {}",
                        metrics.total_divergence, threshold
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Trait for simulation tests
pub trait SimTest {
    /// Name of this test
    fn name(&self) -> &str;

    /// Invariants to check each frame
    fn invariants(&self) -> Vec<Invariant>;

    /// Number of frames to run
    fn run_frames(&self) -> u32;

    /// Description of what this test validates
    fn description(&self) -> &str;
}

/// Test levels as defined in the cleanup protocol
pub mod levels {
    use super::*;

    /// Level 0: FLIP alone - dam break in a box (no sediment, no riffles)
    pub struct Level0DamBreak;

    impl SimTest for Level0DamBreak {
        fn name(&self) -> &str { "Level 0: Dam Break" }

        fn invariants(&self) -> Vec<Invariant> {
            vec![
                Invariant::NoNaN,
                Invariant::NoParticlesInSolid,
                Invariant::ParticleCountStable { tolerance_pct: 0.05 },
                Invariant::MaxVelocityBelow { limit: 20.0 },
            ]
        }

        fn run_frames(&self) -> u32 { 300 }

        fn description(&self) -> &str {
            "Pure FLIP water simulation. A block of water collapses under gravity in a box. \
             Tests basic P2G, pressure solve, G2P cycle."
        }
    }

    /// Level 1: FLIP with sloped floor (tests gravity + BC)
    pub struct Level1SlopedFloor;

    impl SimTest for Level1SlopedFloor {
        fn name(&self) -> &str { "Level 1: Sloped Floor Flow" }

        fn invariants(&self) -> Vec<Invariant> {
            vec![
                Invariant::NoNaN,
                Invariant::NoParticlesInSolid,
                Invariant::ParticleCountStable { tolerance_pct: 0.05 },
                Invariant::MaxVelocityBelow { limit: 15.0 },
            ]
        }

        fn run_frames(&self) -> u32 { 500 }

        fn description(&self) -> &str {
            "Water flows down a sloped floor with continuous inlet. \
             Tests flow acceleration and boundary conditions."
        }
    }

    /// Level 2: FLIP with riffles (tests SDF collision)
    pub struct Level2Riffles;

    impl SimTest for Level2Riffles {
        fn name(&self) -> &str { "Level 2: Flow Over Riffles" }

        fn invariants(&self) -> Vec<Invariant> {
            // Note: No ParticleCountStable for flow-through tests with continuous emission
            vec![
                Invariant::NoNaN,
                Invariant::NoParticlesInSolid,
                Invariant::MaxVelocityBelow { limit: 10.0 },
                Invariant::DivergenceBelow { threshold: 50.0 },
            ]
        }

        fn run_frames(&self) -> u32 { 600 }

        fn description(&self) -> &str {
            "Water flows over riffle obstacles without sediment. \
             Tests SDF-based solid boundary enforcement."
        }
    }

    /// Level 3: FLIP + passive sediment (one-way coupling)
    pub struct Level3PassiveSediment;

    impl SimTest for Level3PassiveSediment {
        fn name(&self) -> &str { "Level 3: Passive Sediment" }

        fn invariants(&self) -> Vec<Invariant> {
            vec![
                Invariant::NoNaN,
                Invariant::ParticleCountStable { tolerance_pct: 0.15 },
                Invariant::DivergenceBelow { threshold: 100.0 },
            ]
        }

        fn run_frames(&self) -> u32 { 600 }

        fn description(&self) -> &str {
            "Sediment particles are transported by water without feedback. \
             Tests one-way coupling (sediment feels water, water ignores sediment)."
        }
    }

    /// Level 4: Sediment settling test
    pub struct Level4SedimentSettling;

    impl SimTest for Level4SedimentSettling {
        fn name(&self) -> &str { "Level 4: Sediment Settling" }

        fn invariants(&self) -> Vec<Invariant> {
            vec![
                Invariant::NoNaN,
                Invariant::ParticleCountStable { tolerance_pct: 0.10 },
            ]
        }

        fn run_frames(&self) -> u32 { 600 }

        fn description(&self) -> &str {
            "Heavy sediment should settle through water. \
             Gold (19.3 g/cm³) should settle faster than gangue (2.7 g/cm³)."
        }
    }

    /// Level 5: Vorticity and suspension
    pub struct Level5VorticitySuspension;

    impl SimTest for Level5VorticitySuspension {
        fn name(&self) -> &str { "Level 5: Vorticity Suspension" }

        fn invariants(&self) -> Vec<Invariant> {
            vec![
                Invariant::NoNaN,
                Invariant::ParticleCountStable { tolerance_pct: 0.15 },
            ]
        }

        fn run_frames(&self) -> u32 { 800 }

        fn description(&self) -> &str {
            "Sediment should be lifted by vorticity in turbulent regions. \
             Tests vorticity-based lift force implementation."
        }
    }

    /// Level 6: Bed formation (sediment stacking)
    pub struct Level6BedFormation;

    impl SimTest for Level6BedFormation {
        fn name(&self) -> &str { "Level 6: Bed Formation" }

        fn invariants(&self) -> Vec<Invariant> {
            vec![
                Invariant::NoNaN,
                Invariant::ParticleCountStable { tolerance_pct: 0.20 },
            ]
        }

        fn run_frames(&self) -> u32 { 1200 }

        fn description(&self) -> &str {
            "Sediment should accumulate and form a bed heightfield. \
             Currently BROKEN: particles don't register in grid cells."
        }
    }

    /// Level 7: Gold separation
    pub struct Level7GoldSeparation;

    impl SimTest for Level7GoldSeparation {
        fn name(&self) -> &str { "Level 7: Gold Separation" }

        fn invariants(&self) -> Vec<Invariant> {
            vec![
                Invariant::NoNaN,
                Invariant::ParticleCountStable { tolerance_pct: 0.20 },
            ]
        }

        fn run_frames(&self) -> u32 { 1800 }

        fn description(&self) -> &str {
            "Gold should concentrate in riffle pockets while gangue washes out. \
             This is the ultimate goal of the simulation."
        }
    }

    /// Level 8: Full sluice operation
    pub struct Level8FullSluice;

    impl SimTest for Level8FullSluice {
        fn name(&self) -> &str { "Level 8: Full Sluice" }

        fn invariants(&self) -> Vec<Invariant> {
            vec![
                Invariant::NoNaN,
                Invariant::ParticleCountStable { tolerance_pct: 0.25 },
            ]
        }

        fn run_frames(&self) -> u32 { 3600 } // 60 seconds at 60 FPS

        fn description(&self) -> &str {
            "Complete sluice operation with continuous feed and exit. \
             Tests steady-state behavior over extended runtime."
        }
    }
}

/// Helper to run a test and collect results
pub fn run_test<T: SimTest>(test: &T) -> TestResult {
    let start = Instant::now();

    // Initialize with default metrics
    let mut metrics = TestMetrics::default();
    let mut failures = Vec::new();

    println!("=== {} ===", test.name());
    println!("{}", test.description());
    println!("Running {} frames...", test.run_frames());

    // Note: Actual simulation would be run here by the example binary
    // This is just the framework structure

    metrics.elapsed_seconds = start.elapsed().as_secs_f32();

    TestResult {
        passed: failures.is_empty(),
        test_name: test.name().to_string(),
        metrics,
        failures,
    }
}
