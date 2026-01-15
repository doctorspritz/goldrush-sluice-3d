//! Unified Simulation Manager for examples.
//!
//! Encapsulates the update loop and synchronization between
//! FLIP (fluid) and DEM (solid) simulations.

use std::path::Path;
use sim3d::{FlipSimulation3D, ClusterSimulation3D, SdfParams, Vec3};
use crate::scenario::{Scenario, SimulationState};
use crate::editor::EditorLayout;
use sim3d::{CellType, TestBox, TestFloor, TestSdfGenerator};

/// Manages the simulation lifecycle and data synchronization.
pub struct SimulationManager {
    pub flip: FlipSimulation3D,
    pub dem: ClusterSimulation3D,
    pub paused: bool,
    pub frame: u32,
}

impl SimulationManager {
    /// Create a new manager with default grid and physics parameters.
    pub fn new(width: usize, height: usize, depth: usize, cell_size: f32) -> Self {
        Self {
            flip: FlipSimulation3D::new(width, height, depth, cell_size),
            dem: ClusterSimulation3D::new(glam::Vec3::new(-10.0, -2.0, -10.0), glam::Vec3::new(20.0, 10.0, 20.0)),
            paused: false,
            frame: 0,
        }
    }

    /// Load a scenario (layout + optional state).
    pub fn load_scenario(&mut self, scenario: &Scenario) {
        // Reset simulations
        let (gw, gh, gd) = (
            self.flip.grid.width,
            self.flip.grid.height,
            self.flip.grid.depth,
        );
        let cs = self.flip.grid.cell_size;

        // If scenario has state, restore it
        if let Some(state) = &scenario.state {
            if let Some(first_flip) = state.flips.first() {
                self.flip = first_flip.clone();
            } else {
                self.flip = FlipSimulation3D::new(gw, gh, gd, cs);
            }

            if let Some(dem_state) = &state.dem {
                self.dem = dem_state.clone();
            } else {
                self.dem = ClusterSimulation3D::new(glam::Vec3::new(-10.0, -2.0, -10.0), glam::Vec3::new(20.0, 10.0, 20.0));
            }
        } else {
            // No state, just reset
            self.flip = FlipSimulation3D::new(gw, gh, gd, cs);
            self.dem = ClusterSimulation3D::new(glam::Vec3::new(-10.0, -2.0, -10.0), glam::Vec3::new(20.0, 10.0, 20.0));
        }



        // Generate SDF from Layout
        if !scenario.layout.test_floors.is_empty() || !scenario.layout.test_boxes.is_empty() {
             let mut gen = TestSdfGenerator::new(
                self.flip.grid.width,
                self.flip.grid.height,
                self.flip.grid.depth,
                self.flip.grid.cell_size,
                Vec3::ZERO,
            );

            for floor in &scenario.layout.test_floors {
                let tf = TestFloor::new(floor.y);
                gen.add_floor(&tf);
            }

            for box_piece in &scenario.layout.test_boxes {
                let tb = TestBox::new(
                    box_piece.position,
                    box_piece.width,
                    box_piece.depth,
                    box_piece.wall_height
                );
                gen.add_box(&tb);
            }

            // Update FLIP Grid SDF & Cell Types
            self.flip.grid.sdf = gen.sdf;

            for i in 0..self.flip.grid.cell_type.len() {
                if self.flip.grid.sdf[i] < 0.0 {
                    self.flip.grid.cell_type[i] = CellType::Solid;
                    self.flip.grid.solid[i] = true;
                }
            }
        }

        self.frame = self.flip.frame;
    }

    /// Update both simulations and synchronize data.
    pub fn update(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        // 1. Update FLIP (Fluid)
        self.flip.update(dt);

        // 2. Synchronize FLIP grid to DEM SDF for collisions
        let sdf_params = SdfParams {
            sdf: &self.flip.grid.sdf,
            grid_width: self.flip.grid.width,
            grid_height: self.flip.grid.height,
            grid_depth: self.flip.grid.depth,
            cell_size: self.flip.grid.cell_size,
            grid_offset: Vec3::ZERO,
        };

        // 3. Update DEM (Solids)
        self.dem.step_with_sdf(dt, &sdf_params);

        self.frame += 1;
    }

    /// Toggle pause.
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// Save current state as a snapshot.
    pub fn create_state(&self) -> SimulationState {
        SimulationState {
            flips: vec![self.flip.clone()],
            dem: Some(self.dem.clone()),
        }
    }
}
