use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Clone, Debug)]
struct Config {
    world_size_m: f32,
    base_cell_size_m: f32,
    tile_resolution: u32,
    lod_levels: u32,
    tile_pool: Vec<u32>,
    update_stride: Vec<u32>,
    activity_thresholds: Vec<f32>,
    activity_sleep_frames: u32,
    camera_radius_tiles: i32,
    activity_hotspots: u32,
    activity_hotspot_radius_m: f32,
    memory_budget_mb: u32,
    buffers_per_cell: u32,
    full_double_buffer: bool,
    frames: u32,
    report_stride: u32,
    enforce_parent_tiles: bool,
    csv_path: String,
    ascii_map_stride: u32,
    ascii_map_lod: u32,
    ascii_map_max_dim: u32,
    image_output_dir: String,
    image_stride: u32,
    image_lod: u32,
    image_scale: u32,
    image_max_dim: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            world_size_m: 2000.0,
            base_cell_size_m: 0.2,
            tile_resolution: 256,
            lod_levels: 4,
            tile_pool: vec![128, 128, 64, 25],
            update_stride: vec![1, 2, 4, 8],
            activity_thresholds: vec![0.7, 0.4, 0.2, 0.0],
            activity_sleep_frames: 120,
            camera_radius_tiles: 3,
            activity_hotspots: 3,
            activity_hotspot_radius_m: 120.0,
            memory_budget_mb: 4096,
            buffers_per_cell: 11,
            full_double_buffer: true,
            frames: 300,
            report_stride: 30,
            enforce_parent_tiles: true,
            csv_path: String::new(),
            ascii_map_stride: 0,
            ascii_map_lod: 0,
            ascii_map_max_dim: 80,
            image_output_dir: String::new(),
            image_stride: 0,
            image_lod: 0,
            image_scale: 8,
            image_max_dim: 200,
        }
    }
}

impl Config {
    fn normalize(&mut self) {
        if self.lod_levels == 0 {
            self.lod_levels = 1;
        }
        let lods = self.lod_levels as usize;
        let pool_fill = self.tile_pool.last().copied().unwrap_or(0);
        let stride_fill = self.update_stride.last().copied().unwrap_or(1);
        let activity_fill = self.activity_thresholds.last().copied().unwrap_or(0.0);
        ensure_len(&mut self.tile_pool, lods, pool_fill);
        ensure_len(
            &mut self.update_stride,
            lods,
            stride_fill,
        );
        ensure_len_f32(
            &mut self.activity_thresholds,
            lods,
            activity_fill,
        );
        self.tile_pool.truncate(lods);
        self.update_stride.truncate(lods);
        self.activity_thresholds.truncate(lods);
        if self.ascii_map_lod >= self.lod_levels {
            self.ascii_map_lod = self.lod_levels.saturating_sub(1);
        }
        if self.image_lod >= self.lod_levels {
            self.image_lod = self.lod_levels.saturating_sub(1);
        }
        if self.csv_path.eq_ignore_ascii_case("none") {
            self.csv_path.clear();
        }
        if self.ascii_map_max_dim == 0 {
            self.ascii_map_max_dim = 1;
        }
        if self.image_scale == 0 {
            self.image_scale = 1;
        }
        if self.image_max_dim == 0 {
            self.image_max_dim = 1;
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
struct TileKey {
    lod: u8,
    tx: i32,
    tz: i32,
}

#[derive(Clone, Debug)]
struct TileState {
    last_active_frame: u64,
    sleep_frames: u32,
    activity: f32,
}

#[derive(Clone, Debug)]
struct LodLevel {
    lod: u8,
    cell_size: f32,
    tile_size_m: f32,
    tiles_x: i32,
    tiles_z: i32,
    update_stride: u32,
    tile_pool: u32,
    activity_threshold: f32,
}

#[derive(Clone, Debug)]
struct ActivityLevel {
    tiles_x: i32,
    tiles_z: i32,
    activity: Vec<f32>,
}

impl ActivityLevel {
    fn idx(&self, tx: i32, tz: i32) -> usize {
        (tz * self.tiles_x + tx) as usize
    }

    fn get(&self, tx: i32, tz: i32) -> f32 {
        if tx < 0 || tz < 0 || tx >= self.tiles_x || tz >= self.tiles_z {
            0.0
        } else {
            self.activity[self.idx(tx, tz)]
        }
    }
}

#[derive(Clone, Debug, Default)]
struct DispatchCounts {
    surface: u64,
    flux: u64,
    depth: u64,
    erosion: u64,
    upsample: u64,
    downsample: u64,
}

#[derive(Clone, Debug)]
struct Hotspot {
    x: f32,
    z: f32,
    radius: f32,
    intensity: f32,
}

struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        // 64-bit LCG constants
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        let v = self.next_u32();
        (v as f32) / (u32::MAX as f32)
    }

    fn range_f32(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

struct PrototypeSim {
    cfg: Config,
    levels: Vec<LodLevel>,
    rng: LcgRng,
    hotspots: Vec<Hotspot>,
    tiles: HashMap<TileKey, TileState>,
    csv: Option<fs::File>,
}

impl PrototypeSim {
    fn new(cfg: Config) -> Self {
        let levels = build_levels(&cfg);
        let mut rng = LcgRng::new(0x1234_5678_9abc_def0);
        let hotspots = (0..cfg.activity_hotspots)
            .map(|_| Hotspot {
                x: rng.range_f32(0.0, cfg.world_size_m),
                z: rng.range_f32(0.0, cfg.world_size_m),
                radius: cfg.activity_hotspot_radius_m,
                intensity: rng.range_f32(0.6, 1.0),
            })
            .collect();
        let mut csv = open_csv(&cfg);
        if let Some(file) = csv.as_mut() {
            if let Err(err) = write_csv_header(file, cfg.lod_levels as usize) {
                eprintln!("CSV header write failed: {err}");
                csv = None;
            }
        }

        Self {
            cfg,
            levels,
            rng,
            hotspots,
            tiles: HashMap::new(),
            csv,
        }
    }

    fn run(&mut self) {
        let total_frames = self.cfg.frames as u64;
        for frame in 0..total_frames {
            let camera = self.camera_position(frame);
            self.update_hotspots();

            let activity_levels = self.build_activity_levels(camera);
            let selection = select_tiles(
                &self.levels,
                &activity_levels,
                &self.cfg,
                camera,
            );
            self.maybe_print_ascii_map(frame, camera, &activity_levels, &selection.selected_set);
            self.maybe_write_ppm(frame, camera, &activity_levels, &selection.selected_set);

            let stats = self.update_tiles(&selection.selected_set, &activity_levels, frame);
            let dispatch = compute_dispatches(
                &self.levels,
                &selection.selected_by_lod,
                frame,
            );
            let mem_mb = estimate_memory_mb(&self.cfg, &selection.selected_by_lod);
            let budget_mb = self.cfg.memory_budget_mb as f32;
            let budget_flag = if mem_mb > budget_mb { "OVER" } else { "OK" };
            self.maybe_write_csv(frame, &selection, &stats, &dispatch, mem_mb);

            if frame == 0 || frame % self.cfg.report_stride as u64 == 0 {
                let totals = selection
                    .selected_by_lod
                    .iter()
                    .enumerate()
                    .map(|(lod, tiles)| (lod, tiles.len()))
                    .collect::<Vec<_>>();

                println!(
                    "frame {:>4} | L0 {:>4}/{:<4} L1 {:>4}/{:<4} L2 {:>4}/{:<4} L3 {:>4}/{:<4} | mem {:>6.1}/{:<6.1}MB {} | new {:>3} evict {:>3} | pass surf {:>4} flux {:>4} depth {:>4} erosion {:>4} up {:>4} down {:>4}",
                    frame,
                    totals.get(0).map(|t| t.1).unwrap_or(0),
                    self.levels.get(0).map(|l| (l.tiles_x * l.tiles_z) as usize).unwrap_or(0),
                    totals.get(1).map(|t| t.1).unwrap_or(0),
                    self.levels.get(1).map(|l| (l.tiles_x * l.tiles_z) as usize).unwrap_or(0),
                    totals.get(2).map(|t| t.1).unwrap_or(0),
                    self.levels.get(2).map(|l| (l.tiles_x * l.tiles_z) as usize).unwrap_or(0),
                    totals.get(3).map(|t| t.1).unwrap_or(0),
                    self.levels.get(3).map(|l| (l.tiles_x * l.tiles_z) as usize).unwrap_or(0),
                    mem_mb,
                    budget_mb,
                    budget_flag,
                    stats.new_tiles,
                    stats.evicted_tiles,
                    dispatch.surface,
                    dispatch.flux,
                    dispatch.depth,
                    dispatch.erosion,
                    dispatch.upsample,
                    dispatch.downsample,
                );
            }
        }
    }

    fn camera_position(&self, frame: u64) -> (f32, f32) {
        let t = (frame as f32) / (self.cfg.frames.max(1) as f32);
        let angle = t * std::f32::consts::TAU;
        let radius = self.cfg.world_size_m * 0.25;
        let center = self.cfg.world_size_m * 0.5;
        let x = center + radius * angle.cos();
        let z = center + radius * angle.sin();
        (x, z)
    }

    fn update_hotspots(&mut self) {
        let drift = self.cfg.activity_hotspot_radius_m * 0.05;
        for hotspot in &mut self.hotspots {
            hotspot.x += self.rng.range_f32(-drift, drift);
            hotspot.z += self.rng.range_f32(-drift, drift);
            hotspot.x = hotspot.x.clamp(0.0, self.cfg.world_size_m);
            hotspot.z = hotspot.z.clamp(0.0, self.cfg.world_size_m);
        }
    }

    fn build_activity_levels(&self, camera: (f32, f32)) -> Vec<ActivityLevel> {
        let mut levels = Vec::with_capacity(self.levels.len());
        if let Some(l0) = self.levels.first() {
            levels.push(build_l0_activity(l0, &self.cfg, camera, &self.hotspots));
        }
        for i in 1..self.levels.len() {
            let grid = &self.levels[i];
            let prev = &levels[i - 1];
            levels.push(downsample_max(prev, grid.tiles_x, grid.tiles_z));
        }
        levels
    }

    fn update_tiles(
        &mut self,
        selected: &HashSet<TileKey>,
        activity_levels: &[ActivityLevel],
        frame: u64,
    ) -> TileStats {
        let mut new_tiles = 0;
        for key in selected.iter() {
            if !self.tiles.contains_key(key) {
                new_tiles += 1;
            }
            let activity = activity_at(activity_levels, *key);
            let entry = self.tiles.entry(*key).or_insert(TileState {
                last_active_frame: frame,
                sleep_frames: 0,
                activity,
            });
            entry.last_active_frame = frame;
            entry.sleep_frames = 0;
            entry.activity = activity;
        }

        let mut evicted_tiles = 0;
        let sleep_limit = self.cfg.activity_sleep_frames;
        self.tiles.retain(|key, state| {
            if selected.contains(key) {
                true
            } else {
                state.sleep_frames = state.sleep_frames.saturating_add(1);
                if state.sleep_frames > sleep_limit {
                    evicted_tiles += 1;
                    false
                } else {
                    true
                }
            }
        });

        TileStats {
            new_tiles,
            evicted_tiles,
        }
    }

    fn maybe_write_csv(
        &mut self,
        frame: u64,
        selection: &Selection,
        stats: &TileStats,
        dispatch: &DispatchCounts,
        mem_mb: f32,
    ) {
        let file = match self.csv.as_mut() {
            Some(file) => file,
            None => return,
        };

        let lods = self.levels.len();
        let mut counts = vec![0usize; lods];
        for (lod, tiles) in selection.selected_by_lod.iter().enumerate() {
            if lod < counts.len() {
                counts[lod] = tiles.len();
            }
        }

        let mut line = String::new();
        line.push_str(&format!("{}", frame));
        for count in counts {
            line.push_str(&format!(",{}", count));
        }
        line.push_str(&format!(
            ",{:.2},{},{},{},{},{},{},{},{}",
            mem_mb,
            stats.new_tiles,
            stats.evicted_tiles,
            dispatch.surface,
            dispatch.flux,
            dispatch.depth,
            dispatch.erosion,
            dispatch.upsample,
            dispatch.downsample,
        ));
        line.push('\n');

        if let Err(err) = file.write_all(line.as_bytes()) {
            eprintln!("CSV write failed: {err}");
            self.csv = None;
        }
    }

    fn maybe_print_ascii_map(
        &self,
        frame: u64,
        camera: (f32, f32),
        activity_levels: &[ActivityLevel],
        selected: &HashSet<TileKey>,
    ) {
        if self.cfg.ascii_map_stride == 0 {
            return;
        }
        if frame % self.cfg.ascii_map_stride as u64 != 0 {
            return;
        }

        let lod = self.cfg.ascii_map_lod as usize;
        let level = match self.levels.get(lod) {
            Some(level) => level,
            None => return,
        };
        let activity = match activity_levels.get(lod) {
            Some(activity) => activity,
            None => return,
        };

        let max_dim = self.cfg.ascii_map_max_dim as i32;
        if activity.tiles_x > max_dim || activity.tiles_z > max_dim {
            println!(
                "map L{} frame {} skipped ({}x{} > max {})",
                lod,
                frame,
                activity.tiles_x,
                activity.tiles_z,
                max_dim
            );
            return;
        }

        let cam_tx = (camera.0 / level.tile_size_m).floor() as i32;
        let cam_tz = (camera.1 / level.tile_size_m).floor() as i32;
        println!(
            "map L{} frame {} ({}x{})",
            lod, frame, activity.tiles_x, activity.tiles_z
        );

        for tz in 0..activity.tiles_z {
            let mut row = String::with_capacity(activity.tiles_x as usize);
            for tx in 0..activity.tiles_x {
                let key = TileKey {
                    lod: level.lod,
                    tx,
                    tz,
                };
                let mut ch = activity_char(activity.get(tx, tz));
                if selected.contains(&key) {
                    ch = '#';
                }
                if tx == cam_tx && tz == cam_tz {
                    ch = 'C';
                }
                row.push(ch);
            }
            println!("{row}");
        }
        println!("legend: C=camera #=selected O/o/*/:/.=activity");
    }

    fn maybe_write_ppm(
        &self,
        frame: u64,
        camera: (f32, f32),
        activity_levels: &[ActivityLevel],
        selected: &HashSet<TileKey>,
    ) {
        if self.cfg.image_stride == 0 {
            return;
        }
        if frame % self.cfg.image_stride as u64 != 0 {
            return;
        }
        let output_dir = self.cfg.image_output_dir.trim();
        if output_dir.is_empty() {
            return;
        }

        let lod = self.cfg.image_lod as usize;
        let level = match self.levels.get(lod) {
            Some(level) => level,
            None => return,
        };
        let activity = match activity_levels.get(lod) {
            Some(activity) => activity,
            None => return,
        };

        let max_dim = self.cfg.image_max_dim as i32;
        if activity.tiles_x > max_dim || activity.tiles_z > max_dim {
            println!(
                "image L{} frame {} skipped ({}x{} > max {})",
                lod,
                frame,
                activity.tiles_x,
                activity.tiles_z,
                max_dim
            );
            return;
        }

        if let Err(err) = fs::create_dir_all(output_dir) {
            eprintln!("Failed to create image output dir '{}': {err}", output_dir);
            return;
        }

        let scale = self.cfg.image_scale as usize;
        let width = activity.tiles_x as usize * scale;
        let height = activity.tiles_z as usize * scale;
        let mut pixels = Vec::with_capacity(width * height * 3);

        let cam_tx = (camera.0 / level.tile_size_m).floor() as i32;
        let cam_tz = (camera.1 / level.tile_size_m).floor() as i32;

        for tz in 0..activity.tiles_z {
            for _ in 0..scale {
                for tx in 0..activity.tiles_x {
                    let key = TileKey {
                        lod: level.lod,
                        tx,
                        tz,
                    };
                    let is_selected = selected.contains(&key);
                    let is_camera = tx == cam_tx && tz == cam_tz;
                    let color = tile_color(activity.get(tx, tz), is_selected, is_camera);
                    for _ in 0..scale {
                        pixels.push(color[0]);
                        pixels.push(color[1]);
                        pixels.push(color[2]);
                    }
                }
            }
        }

        let file_name = format!("frame_{:04}_lod{}.ppm", frame, lod);
        let path = Path::new(output_dir).join(file_name);
        if let Err(err) = write_ppm(&path, width as u32, height as u32, &pixels) {
            eprintln!("Failed to write ppm '{:?}': {err}", path);
        }
    }
}

#[derive(Clone, Debug)]
struct TileStats {
    new_tiles: u32,
    evicted_tiles: u32,
}

struct Selection {
    selected_by_lod: Vec<Vec<TileKey>>,
    selected_set: HashSet<TileKey>,
}

fn build_levels(cfg: &Config) -> Vec<LodLevel> {
    let mut levels = Vec::with_capacity(cfg.lod_levels as usize);
    for lod in 0..cfg.lod_levels {
        let cell_size = cfg.base_cell_size_m * 2.0_f32.powi(lod as i32);
        let tile_size_m = cell_size * cfg.tile_resolution as f32;
        let tiles_x = (cfg.world_size_m / tile_size_m).ceil() as i32;
        let tiles_z = (cfg.world_size_m / tile_size_m).ceil() as i32;
        levels.push(LodLevel {
            lod: lod as u8,
            cell_size,
            tile_size_m,
            tiles_x,
            tiles_z,
            update_stride: cfg.update_stride[lod as usize],
            tile_pool: cfg.tile_pool[lod as usize],
            activity_threshold: cfg.activity_thresholds[lod as usize],
        });
    }
    levels
}

fn build_l0_activity(
    level: &LodLevel,
    cfg: &Config,
    camera: (f32, f32),
    hotspots: &[Hotspot],
) -> ActivityLevel {
    let tiles_x = level.tiles_x;
    let tiles_z = level.tiles_z;
    let mut activity = vec![0.0_f32; (tiles_x * tiles_z) as usize];
    let camera_radius_m = cfg.camera_radius_tiles as f32 * level.tile_size_m;

    for tz in 0..tiles_z {
        for tx in 0..tiles_x {
            let center_x = (tx as f32 + 0.5) * level.tile_size_m;
            let center_z = (tz as f32 + 0.5) * level.tile_size_m;
            let mut a = radial_falloff(distance(camera.0, camera.1, center_x, center_z), camera_radius_m);

            for hotspot in hotspots {
                let d = distance(hotspot.x, hotspot.z, center_x, center_z);
                let h = radial_falloff(d, hotspot.radius) * hotspot.intensity;
                if h > a {
                    a = h;
                }
            }

            activity[(tz * tiles_x + tx) as usize] = a;
        }
    }

    ActivityLevel {
        tiles_x,
        tiles_z,
        activity,
    }
}

fn downsample_max(prev: &ActivityLevel, tiles_x: i32, tiles_z: i32) -> ActivityLevel {
    let mut activity = vec![0.0_f32; (tiles_x * tiles_z) as usize];

    for tz in 0..tiles_z {
        for tx in 0..tiles_x {
            let mut max_val = 0.0_f32;
            for dz in 0..2 {
                for dx in 0..2 {
                    let px = tx * 2 + dx;
                    let pz = tz * 2 + dz;
                    if px < prev.tiles_x && pz < prev.tiles_z {
                        let v = prev.get(px, pz);
                        if v > max_val {
                            max_val = v;
                        }
                    }
                }
            }
            activity[(tz * tiles_x + tx) as usize] = max_val;
        }
    }

    ActivityLevel {
        tiles_x,
        tiles_z,
        activity,
    }
}

fn select_tiles(
    levels: &[LodLevel],
    activity_levels: &[ActivityLevel],
    cfg: &Config,
    camera: (f32, f32),
) -> Selection {
    let mut selected_by_lod: Vec<Vec<TileKey>> = vec![Vec::new(); levels.len()];
    let mut selected_set: HashSet<TileKey> = HashSet::new();

    for (lod, level) in levels.iter().enumerate() {
        let activity = &activity_levels[lod];
        let mut candidates: Vec<(TileKey, f32)> = Vec::new();

        for tz in 0..activity.tiles_z {
            for tx in 0..activity.tiles_x {
                let a = activity.get(tx, tz);
                if a >= level.activity_threshold {
                    candidates.push((
                        TileKey {
                            lod: level.lod,
                            tx,
                            tz,
                        },
                        a,
                    ));
                }
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let limit = level.tile_pool as usize;
        for (key, _) in candidates.into_iter().take(limit) {
            if selected_set.insert(key) {
                selected_by_lod[lod].push(key);
            }
        }
    }

    // Force camera ring at L0
    if let Some(l0) = levels.first() {
        let camera_tx = (camera.0 / l0.tile_size_m).floor() as i32;
        let camera_tz = (camera.1 / l0.tile_size_m).floor() as i32;
        let radius = cfg.camera_radius_tiles;

        for dz in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dz * dz > radius * radius {
                    continue;
                }
                let tx = camera_tx + dx;
                let tz = camera_tz + dz;
                if tx < 0 || tz < 0 || tx >= l0.tiles_x || tz >= l0.tiles_z {
                    continue;
                }
                let key = TileKey {
                    lod: 0,
                    tx,
                    tz,
                };
                if selected_set.insert(key) {
                    selected_by_lod[0].push(key);
                }
            }
        }
    }

    if cfg.enforce_parent_tiles {
        let mut to_add = Vec::new();
        for key in selected_set.iter() {
            let mut current = *key;
            while let Some(parent) = parent_tile(current, levels.len() as u8) {
                if selected_set.contains(&parent) {
                    current = parent;
                    continue;
                }
                to_add.push(parent);
                current = parent;
            }
        }
        for parent in to_add {
            if selected_set.insert(parent) {
                let lod_idx = parent.lod as usize;
                if lod_idx < selected_by_lod.len() {
                    selected_by_lod[lod_idx].push(parent);
                }
            }
        }
    }

    Selection {
        selected_by_lod,
        selected_set,
    }
}

fn parent_tile(key: TileKey, lod_levels: u8) -> Option<TileKey> {
    if key.lod + 1 >= lod_levels {
        return None;
    }
    Some(TileKey {
        lod: key.lod + 1,
        tx: key.tx / 2,
        tz: key.tz / 2,
    })
}

fn compute_dispatches(
    levels: &[LodLevel],
    selected_by_lod: &[Vec<TileKey>],
    frame: u64,
) -> DispatchCounts {
    let mut counts = DispatchCounts::default();

    for (lod, level) in levels.iter().enumerate() {
        if frame % level.update_stride as u64 != 0 {
            continue;
        }
        let tile_count = selected_by_lod.get(lod).map(|v| v.len()).unwrap_or(0) as u64;
        if tile_count == 0 {
            continue;
        }
        counts.surface += tile_count;
        counts.flux += tile_count;
        counts.depth += tile_count;
        counts.erosion += tile_count;
        if level.lod > 0 {
            counts.upsample += tile_count;
        }
        if (level.lod as usize + 1) < levels.len() {
            counts.downsample += tile_count;
        }
    }

    counts
}

fn estimate_memory_mb(cfg: &Config, selected_by_lod: &[Vec<TileKey>]) -> f32 {
    let bytes_per_cell = cfg.buffers_per_cell as u64 * 4 * if cfg.full_double_buffer { 2 } else { 1 };
    let cells_per_tile = (cfg.tile_resolution as u64) * (cfg.tile_resolution as u64);
    let bytes_per_tile = bytes_per_cell * cells_per_tile;

    let total_tiles: u64 = selected_by_lod.iter().map(|v| v.len() as u64).sum();
    let total_bytes = bytes_per_tile * total_tiles;
    total_bytes as f32 / (1024.0 * 1024.0)
}

fn activity_at(levels: &[ActivityLevel], key: TileKey) -> f32 {
    levels
        .get(key.lod as usize)
        .map(|lvl| lvl.get(key.tx, key.tz))
        .unwrap_or(0.0)
}

fn activity_char(value: f32) -> char {
    if value >= 0.9 {
        'O'
    } else if value >= 0.75 {
        'o'
    } else if value >= 0.5 {
        '*'
    } else if value >= 0.25 {
        ':'
    } else if value > 0.0 {
        '.'
    } else {
        ' '
    }
}

fn tile_color(activity: f32, selected: bool, camera: bool) -> [u8; 3] {
    if camera {
        return [40, 220, 60];
    }
    if selected {
        return [220, 80, 40];
    }
    let a = activity.clamp(0.0, 1.0);
    let r = (a * 255.0) as u8;
    let g = (a * a * 255.0) as u8;
    let b = ((1.0 - a) * 180.0 + 50.0) as u8;
    [r, g, b]
}

fn radial_falloff(dist: f32, radius: f32) -> f32 {
    if radius <= 0.0 || dist >= radius {
        0.0
    } else {
        let t = 1.0 - dist / radius;
        t * t
    }
}

fn distance(x0: f32, z0: f32, x1: f32, z1: f32) -> f32 {
    let dx = x1 - x0;
    let dz = z1 - z0;
    (dx * dx + dz * dz).sqrt()
}

fn ensure_len<T: Clone>(vec: &mut Vec<T>, len: usize, fill: T) {
    if vec.len() < len {
        vec.extend(std::iter::repeat(fill).take(len - vec.len()));
    }
}

fn ensure_len_f32(vec: &mut Vec<f32>, len: usize, fill: f32) {
    if vec.len() < len {
        vec.extend(std::iter::repeat(fill).take(len - vec.len()));
    }
}

fn open_csv(cfg: &Config) -> Option<fs::File> {
    let path = cfg.csv_path.trim();
    if path.is_empty() {
        return None;
    }
    match fs::File::create(path) {
        Ok(file) => Some(file),
        Err(err) => {
            eprintln!("Failed to create CSV file '{}': {err}", path);
            None
        }
    }
}

fn write_csv_header(file: &mut fs::File, lod_levels: usize) -> std::io::Result<()> {
    let mut header = String::from("frame");
    for lod in 0..lod_levels {
        header.push_str(&format!(",l{}", lod));
    }
    header.push_str(
        ",mem_mb,new_tiles,evicted_tiles,pass_surface,pass_flux,pass_depth,pass_erosion,pass_upsample,pass_downsample\n",
    );
    file.write_all(header.as_bytes())
}

fn write_ppm(path: &Path, width: u32, height: u32, pixels: &[u8]) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    let header = format!("P6\n{} {}\n255\n", width, height);
    file.write_all(header.as_bytes())?;
    file.write_all(pixels)?;
    Ok(())
}

fn load_config(path: &Path) -> Config {
    match fs::read_to_string(path) {
        Ok(text) => match parse_config(&text) {
            Ok(mut cfg) => {
                cfg.normalize();
                cfg
            }
            Err(err) => {
                eprintln!("Config parse failed ({:?}): {}", path, err);
                Config::default()
            }
        },
        Err(_) => Config::default(),
    }
}

fn parse_config(text: &str) -> Result<Config, String> {
    let mut cfg = Config::default();

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("line {}: expected key = value", line_idx + 1))?;
        let key = key.trim();
        let value = value.trim();

        match key {
            "world_size_m" => cfg.world_size_m = parse_f32(value)?,
            "base_cell_size_m" => cfg.base_cell_size_m = parse_f32(value)?,
            "tile_resolution" => cfg.tile_resolution = parse_u32(value)?,
            "lod_levels" => cfg.lod_levels = parse_u32(value)?,
            "tile_pool" => cfg.tile_pool = parse_list_u32(value)?,
            "update_stride" => cfg.update_stride = parse_list_u32(value)?,
            "activity_thresholds" => cfg.activity_thresholds = parse_list_f32(value)?,
            "activity_sleep_frames" => cfg.activity_sleep_frames = parse_u32(value)?,
            "camera_radius_tiles" => cfg.camera_radius_tiles = parse_i32(value)?,
            "activity_hotspots" => cfg.activity_hotspots = parse_u32(value)?,
            "activity_hotspot_radius_m" => cfg.activity_hotspot_radius_m = parse_f32(value)?,
            "memory_budget_mb" => cfg.memory_budget_mb = parse_u32(value)?,
            "buffers_per_cell" => cfg.buffers_per_cell = parse_u32(value)?,
            "full_double_buffer" => cfg.full_double_buffer = parse_bool(value)?,
            "frames" => cfg.frames = parse_u32(value)?,
            "report_stride" => cfg.report_stride = parse_u32(value)?,
            "enforce_parent_tiles" => cfg.enforce_parent_tiles = parse_bool(value)?,
            "csv_path" => cfg.csv_path = parse_string(value)?,
            "ascii_map_stride" => cfg.ascii_map_stride = parse_u32(value)?,
            "ascii_map_lod" => cfg.ascii_map_lod = parse_u32(value)?,
            "ascii_map_max_dim" => cfg.ascii_map_max_dim = parse_u32(value)?,
            "image_output_dir" => cfg.image_output_dir = parse_string(value)?,
            "image_stride" => cfg.image_stride = parse_u32(value)?,
            "image_lod" => cfg.image_lod = parse_u32(value)?,
            "image_scale" => cfg.image_scale = parse_u32(value)?,
            "image_max_dim" => cfg.image_max_dim = parse_u32(value)?,
            _ => {
                return Err(format!("line {}: unknown key '{}'", line_idx + 1, key));
            }
        }
    }

    Ok(cfg)
}

fn parse_u32(value: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|_| format!("invalid u32: {}", value))
}

fn parse_i32(value: &str) -> Result<i32, String> {
    value
        .parse::<i32>()
        .map_err(|_| format!("invalid i32: {}", value))
}

fn parse_f32(value: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .map_err(|_| format!("invalid f32: {}", value))
}

fn parse_bool(value: &str) -> Result<bool, String> {
    match value.to_ascii_lowercase().as_str() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("invalid bool: {}", value)),
    }
}

fn parse_string(value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    let unquoted = if let Some(stripped) = trimmed.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        stripped
    } else if let Some(stripped) = trimmed
        .strip_prefix('\'')
        .and_then(|s| s.strip_suffix('\''))
    {
        stripped
    } else {
        trimmed
    };
    Ok(unquoted.to_string())
}

fn parse_list_u32(value: &str) -> Result<Vec<u32>, String> {
    value
        .split(',')
        .map(|v| parse_u32(v.trim()))
        .collect()
}

fn parse_list_f32(value: &str) -> Result<Vec<f32>, String> {
    value
        .split(',')
        .map(|v| parse_f32(v.trim()))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let config_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("config.toml");
    let cfg = load_config(Path::new(config_path));
    let mut sim = PrototypeSim::new(cfg);
    sim.run();
}
