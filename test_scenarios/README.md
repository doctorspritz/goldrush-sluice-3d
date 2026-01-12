# Physics Test Scenarios for Visual Inspection

These scenarios correspond to the automated physics tests in `crates/game/tests/physics_validation.rs`.

## Usage

Load a scenario in washplant_editor:
```bash
cargo run -p game --example washplant_editor --release -- test_scenarios/<scenario>.json
```

Press `P` to start the simulation and observe particle behavior.

## Scenarios

| # | File | Test | What to Observe |
|---|------|------|-----------------|
| 01 | `01_floor_collision.json` | `test_dem_floor_collision` | Single particle drops onto floor, bounces, settles |
| 02 | `02_wall_collision.json` | `test_dem_wall_collision` | Particles bounce off gutter walls |
| 03 | `03_density_separation.json` | `test_dem_density_separation` | Gold (yellow) settles below sand (brown) |
| 04 | `04_settling_time.json` | `test_dem_settling_time` | Particle from height settles within ~5 seconds |
| 05 | `05_flow_direction.json` | `test_fluid_flow_direction` | Particles flow downhill on angled gutter |
| 06 | `06_pool_equilibrium.json` | `test_fluid_pool_equilibrium` | Particle comes to rest in flat pool |
| 07 | `07_wall_containment.json` | `test_fluid_wall_containment` | Particles with lateral velocity stay within walls |
| 08 | `08_sediment_settling.json` | `test_sediment_settling` | Sediment particles sink to floor |
| 09 | `09_sediment_advection.json` | `test_sediment_advection` | Particles flow downstream on slope |
| 10 | `10_riffle_capture.json` | `test_sluice_riffle_capture` | Gold captured behind riffles, sand washes out |

## Quick Test All

Run through all scenarios:
```bash
for f in test_scenarios/*.json; do
  echo "=== $f ==="
  cargo run -p game --example washplant_editor --release -- "$f"
done
```

## Controls Reminder

- `P` - Play/Pause simulation
- `WASD/QE` - Move camera
- `Mouse drag` - Rotate view
- `Scroll` - Zoom
- `Escape` - Exit
