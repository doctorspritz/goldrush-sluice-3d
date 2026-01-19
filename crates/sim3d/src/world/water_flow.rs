//! Shallow water equation solver for water flow simulation.
//!
//! This module implements the shallow water equations (SWE) to simulate water flow
//! across the terrain grid. Both coarse and fine regions use similar physics:
//! - Gravity-driven flow based on water surface gradient
//! - Manning friction to model bed roughness
//! - Flux limiting to ensure mass conservation
//! - Volume update to propagate height changes

use super::{FineRegion, World};

impl World {
    /// Update water flow using shallow water equations.
    ///
    /// This is the main coarse-grid water flow solver. It:
    /// 1. Updates velocity field based on pressure gradients and friction
    /// 2. Limits fluxes to prevent overspill
    /// 3. Updates water surface heights
    /// 4. Applies open boundary conditions if enabled
    pub fn update_water_flow(&mut self, dt: f32) {
        let g = self.params.gravity;
        let n = self.params.manning_n;
        let n2 = n * n;
        let width = self.width;
        let depth = self.depth;
        let cell_size = self.cell_size;
        let cell_area = cell_size * cell_size;

        // Ensure water surface >= ground
        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);
                let ground = self.ground_height(x, z);
                if self.water_surface[idx] < ground {
                    self.water_surface[idx] = ground;
                }
            }
        }

        let max_velocity = cell_size / dt * 0.5;

        // Update X velocities
        for z in 0..depth {
            for x in 1..width {
                let flow_idx = self.flow_x_idx(x, z);
                let idx_l = self.idx(x - 1, z);
                let idx_r = self.idx(x, z);

                let h_l = self.water_surface[idx_l];
                let h_r = self.water_surface[idx_r];

                let depth_l = self.water_depth(x - 1, z);
                let depth_r = self.water_depth(x, z);

                if depth_l < 0.001 && depth_r < 0.001 {
                    self.water_flow_x[flow_idx] = 0.0;
                    continue;
                }

                let gradient = (h_l - h_r) / cell_size;
                let avg_depth = 0.5 * (depth_l + depth_r).max(0.01); // Avoid division by zero

                // Apply gravity
                let mut vel = self.water_flow_x[flow_idx] + g * gradient * dt;

                // Apply Manning friction (implicit method for stability)
                // Friction coefficient: g * n² / R^(4/3) where R ≈ depth for wide channel
                let friction_coeff = g * n2 / avg_depth.powf(4.0 / 3.0);
                let friction_factor = 1.0 + friction_coeff * vel.abs() * dt;
                vel /= friction_factor;

                self.water_flow_x[flow_idx] = vel.clamp(-max_velocity, max_velocity);
            }

            // Boundary velocities
            let left_idx = self.flow_x_idx(0, z);
            let right_idx = self.flow_x_idx(width, z);
            self.water_flow_x[left_idx] = 0.0;
            self.water_flow_x[right_idx] = 0.0;
        }

        // Update Z velocities
        for z in 1..depth {
            for x in 0..width {
                let flow_idx = self.flow_z_idx(x, z);
                let idx_b = self.idx(x, z - 1);
                let idx_f = self.idx(x, z);

                let h_b = self.water_surface[idx_b];
                let h_f = self.water_surface[idx_f];

                let depth_b = self.water_depth(x, z - 1);
                let depth_f = self.water_depth(x, z);

                if depth_b < 0.001 && depth_f < 0.001 {
                    self.water_flow_z[flow_idx] = 0.0;
                    continue;
                }

                let gradient = (h_b - h_f) / cell_size;
                let avg_depth = 0.5 * (depth_b + depth_f).max(0.01);

                // Apply gravity
                let mut vel = self.water_flow_z[flow_idx] + g * gradient * dt;

                // Apply Manning friction
                let friction_coeff = g * n2 / avg_depth.powf(4.0 / 3.0);
                let friction_factor = 1.0 + friction_coeff * vel.abs() * dt;
                vel /= friction_factor;

                self.water_flow_z[flow_idx] = vel.clamp(-max_velocity, max_velocity);
            }
        }

        // Boundary Z velocities
        for x in 0..width {
            let back_idx = self.flow_z_idx(x, 0);
            let front_idx = self.flow_z_idx(x, depth);
            self.water_flow_z[back_idx] = 0.0;
            self.water_flow_z[front_idx] = 0.0;
        }

        // Flux limiting: ensure no cell outputs more volume than it has
        self.collapse_deltas.fill(1.0); // Reuse buffer for limits

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);
                let depth_here = self.water_depth(x, z);

                let available_volume = depth_here * cell_area;
                let mut total_outflow_volume = 0.0;

                // Right (x+1)
                let v_right = self.water_flow_x[self.flow_x_idx(x + 1, z)];
                if v_right > 0.0 {
                    let depth_right = if x + 1 < width {
                        self.water_depth(x + 1, z)
                    } else {
                        depth_here
                    };
                    let flux = v_right * 0.5 * (depth_here + depth_right) * cell_size * dt;
                    total_outflow_volume += flux;
                }

                // Left (x)
                let v_left = self.water_flow_x[self.flow_x_idx(x, z)];
                if v_left < 0.0 {
                    let depth_left = if x > 0 {
                        self.water_depth(x - 1, z)
                    } else {
                        depth_here
                    };
                    let flux = -v_left * 0.5 * (depth_left + depth_here) * cell_size * dt;
                    total_outflow_volume += flux;
                }

                // Front (z+1)
                let v_front = self.water_flow_z[self.flow_z_idx(x, z + 1)];
                if v_front > 0.0 {
                    let depth_front = if z + 1 < depth {
                        self.water_depth(x, z + 1)
                    } else {
                        depth_here
                    };
                    let flux = v_front * 0.5 * (depth_here + depth_front) * cell_size * dt;
                    total_outflow_volume += flux;
                }

                // Back (z)
                let v_back = self.water_flow_z[self.flow_z_idx(x, z)];
                if v_back < 0.0 {
                    let depth_back = if z > 0 {
                        self.water_depth(x, z - 1)
                    } else {
                        depth_here
                    };
                    let flux = -v_back * 0.5 * (depth_back + depth_here) * cell_size * dt;
                    total_outflow_volume += flux;
                }

                if total_outflow_volume > available_volume {
                    self.collapse_deltas[idx] = available_volume / total_outflow_volume;
                }
            }
        }

        // Apply Limiters to Flow
        for z in 0..depth {
            for x in 0..width + 1 {
                // Flow X faces
                let flow_idx = self.flow_x_idx(x, z);
                let vel = self.water_flow_x[flow_idx];
                if vel > 0.0 && x > 0 {
                    let donor_idx = self.idx(x - 1, z);
                    self.water_flow_x[flow_idx] *= self.collapse_deltas[donor_idx];
                } else if vel < 0.0 && x < width {
                    let donor_idx = self.idx(x, z);
                    self.water_flow_x[flow_idx] *= self.collapse_deltas[donor_idx];
                }
            }
        }
        for z in 0..depth + 1 {
            // Flow Z faces
            for x in 0..width {
                let flow_idx = self.flow_z_idx(x, z);
                let vel = self.water_flow_z[flow_idx];
                if vel > 0.0 && z > 0 {
                    let donor_idx = self.idx(x, z - 1);
                    self.water_flow_z[flow_idx] *= self.collapse_deltas[donor_idx];
                } else if vel < 0.0 && z < depth {
                    let donor_idx = self.idx(x, z);
                    self.water_flow_z[flow_idx] *= self.collapse_deltas[donor_idx];
                }
            }
        }

        // Volume Update
        // Reuse collapse_deltas for height changes to verify conservation (avoid race condition)
        self.collapse_deltas.fill(0.0);

        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);

                let depth_here = self.water_depth(x, z);
                let depth_left = if x > 0 {
                    self.water_depth(x - 1, z)
                } else {
                    depth_here
                };
                let depth_right = if x + 1 < width {
                    self.water_depth(x + 1, z)
                } else {
                    depth_here
                };
                let depth_back = if z > 0 {
                    self.water_depth(x, z - 1)
                } else {
                    depth_here
                };
                let depth_front = if z + 1 < depth {
                    self.water_depth(x, z + 1)
                } else {
                    depth_here
                };

                let flux_left =
                    self.water_flow_x[self.flow_x_idx(x, z)] * 0.5 * (depth_left + depth_here);
                let flux_right =
                    self.water_flow_x[self.flow_x_idx(x + 1, z)] * 0.5 * (depth_right + depth_here);
                let flux_back =
                    self.water_flow_z[self.flow_z_idx(x, z)] * 0.5 * (depth_back + depth_here);
                let flux_front =
                    self.water_flow_z[self.flow_z_idx(x, z + 1)] * 0.5 * (depth_front + depth_here);

                let volume_change =
                    (flux_left - flux_right + flux_back - flux_front) * cell_size * dt;

                let height_change = volume_change / cell_area;
                self.collapse_deltas[idx] = height_change;
            }
        }

        // Apply height changes
        for z in 0..depth {
            for x in 0..width {
                let idx = self.idx(x, z);
                self.water_surface[idx] += self.collapse_deltas[idx];
                let ground = self.ground_height(x, z);
                self.water_surface[idx] = self.water_surface[idx].max(ground);
            }
        }

        // Open Boundary Condition: Edges are sinks (water flow off map)
        if self.params.open_boundaries {
            for z in 0..depth {
                let idx_left = self.idx(0, z);
                let idx_right = self.idx(width - 1, z);
                self.water_surface[idx_left] = self.ground_height(0, z);
                self.water_surface[idx_right] = self.ground_height(width - 1, z);
            }
            for x in 0..width {
                let idx_back = self.idx(x, 0);
                let idx_front = self.idx(x, depth - 1);
                self.water_surface[idx_back] = self.ground_height(x, 0);
                self.water_surface[idx_front] = self.ground_height(x, depth - 1);
            }
        }
    }
}

impl FineRegion {
    /// Update water flow using simplified shallow water equations.
    ///
    /// This is the fine-grid water flow solver. It uses a simpler approach:
    /// - Linear damping instead of Manning friction for stability
    /// - Simplified flux limiting (no limiting for now)
    /// - Identical volume propagation as coarse grid
    pub fn update_water_flow(fine: &mut FineRegion, dt: f32, gravity: f32, damping_param: f32) {
        let g = gravity;
        let damping = 1.0 - damping_param;
        let width = fine.width;
        let depth = fine.depth;
        let cell_size = fine.cell_size;
        let cell_area = cell_size * cell_size;

        // Ensure water surface >= ground
        for z in 0..depth {
            for x in 0..width {
                let idx = fine.idx(x, z);
                let ground = fine.ground_height(x, z);
                if fine.water_surface[idx] < ground {
                    fine.water_surface[idx] = ground;
                }
            }
        }

        let max_velocity = cell_size / dt * 0.5;

        // Update X velocities
        for z in 0..depth {
            for x in 1..width {
                let flow_idx = fine.flow_x_idx(x, z);
                let idx_l = fine.idx(x - 1, z);
                let idx_r = fine.idx(x, z);

                let h_l = fine.water_surface[idx_l];
                let h_r = fine.water_surface[idx_r];

                let depth_l = fine.water_depth(x - 1, z);
                let depth_r = fine.water_depth(x, z);

                if depth_l < 0.001 && depth_r < 0.001 {
                    fine.water_flow_x[flow_idx] = 0.0;
                    continue;
                }

                let gradient = (h_l - h_r) / cell_size;
                fine.water_flow_x[flow_idx] += g * gradient * dt;
                fine.water_flow_x[flow_idx] *= damping;
                fine.water_flow_x[flow_idx] =
                    fine.water_flow_x[flow_idx].clamp(-max_velocity, max_velocity);
            }

            // Boundary velocities
            let idx0 = fine.flow_x_idx(0, z);
            let idx_width = fine.flow_x_idx(width, z);
            fine.water_flow_x[idx0] = 0.0;
            fine.water_flow_x[idx_width] = 0.0;
        }

        // Update Z velocities
        for z in 1..depth {
            for x in 0..width {
                let flow_idx = fine.flow_z_idx(x, z);
                let idx_b = fine.idx(x, z - 1);
                let idx_f = fine.idx(x, z);

                let h_b = fine.water_surface[idx_b];
                let h_f = fine.water_surface[idx_f];

                let depth_b = fine.water_depth(x, z - 1);
                let depth_f = fine.water_depth(x, z);

                if depth_b < 0.001 && depth_f < 0.001 {
                    fine.water_flow_z[flow_idx] = 0.0;
                    continue;
                }

                let gradient = (h_b - h_f) / cell_size;
                fine.water_flow_z[flow_idx] += g * gradient * dt;
                fine.water_flow_z[flow_idx] *= damping;
                fine.water_flow_z[flow_idx] =
                    fine.water_flow_z[flow_idx].clamp(-max_velocity, max_velocity);
            }
        }

        // Boundary Z velocities
        for x in 0..width {
            let idx0 = fine.flow_z_idx(x, 0);
            let idx_depth = fine.flow_z_idx(x, depth);
            fine.water_flow_z[idx0] = 0.0;
            fine.water_flow_z[idx_depth] = 0.0;
        }

        // Volume update (simplified - no flux limiting for now)
        fine.collapse_deltas.fill(0.0);

        for z in 0..depth {
            for x in 0..width {
                let idx = z * width + x;

                let depth_here = fine.water_depth(x, z);
                let depth_left = if x > 0 { fine.water_depth(x - 1, z) } else { depth_here };
                let depth_right = if x + 1 < width { fine.water_depth(x + 1, z) } else { depth_here };
                let depth_back = if z > 0 { fine.water_depth(x, z - 1) } else { depth_here };
                let depth_front = if z + 1 < depth { fine.water_depth(x, z + 1) } else { depth_here };

                // Pre-compute flow indices to avoid borrow checker issues
                let flow_x_left = z * (width + 1) + x;
                let flow_x_right = z * (width + 1) + x + 1;
                let flow_z_back = z * width + x;
                let flow_z_front = (z + 1) * width + x;

                let flux_left = fine.water_flow_x[flow_x_left] * 0.5 * (depth_left + depth_here);
                let flux_right = fine.water_flow_x[flow_x_right] * 0.5 * (depth_right + depth_here);
                let flux_back = fine.water_flow_z[flow_z_back] * 0.5 * (depth_back + depth_here);
                let flux_front = fine.water_flow_z[flow_z_front] * 0.5 * (depth_front + depth_here);

                let volume_change = (flux_left - flux_right + flux_back - flux_front) * cell_size * dt;
                let height_change = volume_change / cell_area;
                fine.collapse_deltas[idx] = height_change;
            }
        }

        // Apply height changes
        for z in 0..depth {
            for x in 0..width {
                let idx = z * width + x;
                fine.water_surface[idx] += fine.collapse_deltas[idx];
                let ground = fine.ground_height(x, z);
                fine.water_surface[idx] = fine.water_surface[idx].max(ground);
            }
        }
    }
}
