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
    image_composite: bool,
    thrash_window: u32,
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
            image_composite: false,
            thrash_window: 60,
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
        if self.thrash_window == 0 {
            self.thrash_window = 1;
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

#[derive(Clone, Debug, Default)]
struct ThrashTracker {
    window: usize,
    index: usize,
    entries: Vec<ThrashEntry>,
    sum_new: u64,
    sum_evicted: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ThrashEntry {
    new_tiles: u32,
    evicted_tiles: u32,
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

impl ThrashTracker {
    fn new(window: usize) -> Self {
        let window = window.max(1);
        Self {
            window,
            index: 0,
            entries: vec![ThrashEntry::default(); window],
            sum_new: 0,
            sum_evicted: 0,
        }
    }

    fn push(&mut self, entry: ThrashEntry) {
        let old = self.entries[self.index];
        self.sum_new = self.sum_new.saturating_sub(old.new_tiles as u64);
        self.sum_evicted = self.sum_evicted.saturating_sub(old.evicted_tiles as u64);

        self.entries[self.index] = entry;
        self.sum_new += entry.new_tiles as u64;
        self.sum_evicted += entry.evicted_tiles as u64;

        self.index = (self.index + 1) % self.window;
    }

    fn per_frame(&self) -> f32 {
        (self.sum_new + self.sum_evicted) as f32 / self.window as f32
    }

    fn ratio(&self, active_tiles: usize) -> f32 {
        if active_tiles == 0 {
            0.0
        } else {
            self.per_frame() / active_tiles as f32
        }
    }
}

struct PrototypeSim {
    cfg: Config,
    levels: Vec<LodLevel>,
    rng: LcgRng,
    hotspots: Vec<Hotspot>,
    tiles: HashMap<TileKey, TileState>,
    csv: Option<fs::File>,
    thrash: ThrashTracker,
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
        let thrash = ThrashTracker::new(cfg.thrash_window as usize);

        Self {
            cfg,
            levels,
            rng,
            hotspots,
            tiles: HashMap::new(),
            csv,
            thrash,
        }
    }

    fn run(&mut self) {
        let total_frames = self.cfg.frames as u64;
        for frame in 0..total_frames {
            let camera = self.camera_position(frame);
            self.update_hotspots();

            let activity_levels = self.build_activity_levels(camera);
            let mut selection = select_tiles(
                &self.levels,
                &activity_levels,
                &self.cfg,
                camera,
            );
            let budget = apply_memory_budget(
                &mut selection,
                &self.levels,
                &self.cfg,
                &self.tiles,
                &activity_levels,
                frame,
            );
            self.maybe_print_ascii_map(frame, camera, &activity_levels, &selection.selected_set);
            self.maybe_write_ppm(frame, camera, &activity_levels, &selection.selected_set);

            let stats = self.update_tiles(&selection.selected_set, &activity_levels, frame);
            let thrash_entry = ThrashEntry {
                new_tiles: stats.new_tiles,
                evicted_tiles: stats.evicted_tiles + budget.evicted_tiles as u32,
            };
            self.thrash.push(thrash_entry);
            let thrash_per_frame = self.thrash.per_frame();
            let thrash_ratio = self.thrash.ratio(selection.selected_set.len());
            let dispatch = compute_dispatches(
                &self.levels,
                &selection.selected_by_lod,
                frame,
            );
            let mem_mb = estimate_memory_mb(&self.cfg, &selection.selected_by_lod);
            let budget_mb = self.cfg.memory_budget_mb as f32;
            let mem_flag = if mem_mb > budget_mb { "OVER" } else { "OK" };
            self.maybe_write_csv(
                frame,
                &selection,
                &stats,
                &dispatch,
                mem_mb,
                &budget,
                thrash_per_frame,
                thrash_ratio,
            );

            if frame == 0 || frame % self.cfg.report_stride as u64 == 0 {
                let totals = selection
                    .selected_by_lod
                    .iter()
                    .enumerate()
                    .map(|(lod, tiles)| (lod, tiles.len()))
                    .collect::<Vec<_>>();
                let budget_flag = if budget.blocked { "BLOCKED" } else { "OK" };

                println!(
                    "frame {:>4} | L0 {:>4}/{:<4} L1 {:>4}/{:<4} L2 {:>4}/{:<4} L3 {:>4}/{:<4} | mem {:>6.1}/{:<6.1}MB {} | new {:>3} evict {:>3} budget {:>3} {:>7} | thrash {:>5.1} ({:.3}) | pass surf {:>4} flux {:>4} depth {:>4} erosion {:>4} up {:>4} down {:>4}",
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
                    mem_flag,
                    stats.new_tiles,
                    stats.evicted_tiles,
                    budget.evicted_tiles,
                    budget_flag,
                    thrash_per_frame,
                    thrash_ratio,
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
        budget: &BudgetStats,
        thrash_per_frame: f32,
        thrash_ratio: f32,
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
            ",{:.2},{},{},{},{},{},{:.3},{:.5},{},{},{},{},{},{}",
            mem_mb,
            stats.new_tiles,
            stats.evicted_tiles,
            budget.budget_tiles,
            budget.evicted_tiles,
            if budget.blocked { 1 } else { 0 },
            thrash_per_frame,
            thrash_ratio,
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

        if let Err(err) = fs::create_dir_all(output_dir) {
            eprintln!("Failed to create image output dir '{}': {err}", output_dir);
            return;
        }

        let scale = self.cfg.image_scale as usize;
        let max_dim = self.cfg.image_max_dim as i32;
        let (width, height, pixels) =
            match render_lod_pixels(level, activity, selected, camera, scale, max_dim) {
                Some(data) => data,
                None => {
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
            };

        let file_name = format!("frame_{:04}_lod{}.ppm", frame, lod);
        let path = Path::new(output_dir).join(file_name);
        if let Err(err) = write_ppm(&path, width as u32, height as u32, &pixels) {
            eprintln!("Failed to write ppm '{:?}': {err}", path);
        }

        if self.cfg.image_composite {
            self.maybe_write_composite(frame, camera, activity_levels, selected, output_dir);
        }
    }

    fn maybe_write_composite(
        &self,
        frame: u64,
        camera: (f32, f32),
        activity_levels: &[ActivityLevel],
        selected: &HashSet<TileKey>,
        output_dir: &str,
    ) {
        let scale = self.cfg.image_scale as usize;
        let max_dim = self.cfg.image_max_dim as i32;
        let mut rendered = Vec::with_capacity(self.levels.len());
        for (lod, level) in self.levels.iter().enumerate() {
            let activity = match activity_levels.get(lod) {
                Some(activity) => activity,
                None => continue,
            };
            let data = render_lod_pixels(level, activity, selected, camera, scale, max_dim);
            rendered.push(data);
        }

        let mut max_w = 0usize;
        let mut max_h = 0usize;
        for entry in rendered.iter().flatten() {
            max_w = max_w.max(entry.0 as usize);
            max_h = max_h.max(entry.1 as usize);
        }
        if max_w == 0 || max_h == 0 {
            return;
        }

        let lod_count = self.levels.len();
        let grid_cols = (lod_count as f32).sqrt().ceil() as usize;
        let grid_rows = (lod_count + grid_cols - 1) / grid_cols;
        let width = max_w * grid_cols;
        let height = max_h * grid_rows;
        let mut pixels = vec![0u8; width * height * 3];

        for (index, entry) in rendered.into_iter().enumerate() {
            let (w, h, data) = match entry {
                Some(data) => data,
                None => continue,
            };
            let col = index % grid_cols;
            let row = index / grid_cols;
            let offset_x = col * max_w;
            let offset_y = row * max_h;
            blit_pixels(
                &mut pixels,
                width,
                height,
                data,
                w as usize,
                h as usize,
                offset_x,
                offset_y,
            );
        }

        let file_name = format!("frame_{:04}_composite.ppm", frame);
        let path = Path::new(output_dir).join(file_name);
        if let Err(err) = write_ppm(&path, width as u32, height as u32, &pixels) {
            eprintln!("Failed to write composite ppm '{:?}': {err}", path);
        }
    }
}

#[derive(Clone, Debug)]
struct TileStats {
    new_tiles: u32,
    evicted_tiles: u32,
}

#[derive(Clone, Debug, Default)]
struct BudgetStats {
    budget_tiles: usize,
    evicted_tiles: usize,
    blocked: bool,
}

struct Selection {
    selected_by_lod: Vec<Vec<TileKey>>,
    selected_set: HashSet<TileKey>,
    pinned_set: HashSet<TileKey>,
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
    let mut pinned_set: HashSet<TileKey> = HashSet::new();

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
                pinned_set.insert(key);
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
        pinned_set,
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

fn apply_memory_budget(
    selection: &mut Selection,
    levels: &[LodLevel],
    cfg: &Config,
    tiles: &HashMap<TileKey, TileState>,
    activity_levels: &[ActivityLevel],
    frame: u64,
) -> BudgetStats {
    let bytes_per_tile = bytes_per_tile(cfg);
    if bytes_per_tile == 0 {
        return BudgetStats {
            budget_tiles: 0,
            evicted_tiles: 0,
            blocked: true,
        };
    }
    let budget_tiles = ((cfg.memory_budget_mb as u64) * 1024 * 1024 / bytes_per_tile)
        .max(1) as usize;
    let mut stats = BudgetStats {
        budget_tiles,
        evicted_tiles: 0,
        blocked: false,
    };

    if selection.selected_set.len() <= budget_tiles {
        return stats;
    }

    let lod_levels = levels.len() as u8;
    let protected_set = if cfg.enforce_parent_tiles {
        extend_with_parents(&selection.pinned_set, lod_levels)
    } else {
        selection.pinned_set.clone()
    };

    if protected_set.len() > budget_tiles {
        stats.blocked = true;
        return stats;
    }

    while selection.selected_set.len() > budget_tiles {
        let required_parents = if cfg.enforce_parent_tiles {
            required_parents_of_set(&selection.selected_set, lod_levels)
        } else {
            HashSet::new()
        };
        let mut candidates: Vec<TileKey> = selection
            .selected_set
            .iter()
            .copied()
            .filter(|key| !protected_set.contains(key))
            .filter(|key| !required_parents.contains(key))
            .collect();

        if candidates.is_empty() {
            candidates = selection
                .selected_set
                .iter()
                .copied()
                .filter(|key| !protected_set.contains(key))
                .collect();
        }

        if candidates.is_empty() {
            stats.blocked = true;
            break;
        }

        candidates.sort_by(|a, b| {
            let pa = tile_priority(*a, activity_levels, tiles, frame, lod_levels);
            let pb = tile_priority(*b, activity_levels, tiles, frame, lod_levels);
            pa.partial_cmp(&pb).unwrap_or(Ordering::Equal)
        });

        let victim = candidates[0];
        let removed = evict_subtree(&mut selection.selected_set, victim, lod_levels);
        stats.evicted_tiles += removed;

        if removed == 0 {
            stats.blocked = true;
            break;
        }
    }

    rebuild_selected_by_lod(selection, lod_levels as usize);
    stats
}

fn bytes_per_tile(cfg: &Config) -> u64 {
    let bytes_per_cell = cfg.buffers_per_cell as u64 * 4 * if cfg.full_double_buffer { 2 } else { 1 };
    let cells_per_tile = (cfg.tile_resolution as u64) * (cfg.tile_resolution as u64);
    bytes_per_cell.saturating_mul(cells_per_tile)
}

fn rebuild_selected_by_lod(selection: &mut Selection, lod_levels: usize) {
    selection.selected_by_lod = vec![Vec::new(); lod_levels];
    for key in selection.selected_set.iter().copied() {
        let idx = key.lod as usize;
        if idx < selection.selected_by_lod.len() {
            selection.selected_by_lod[idx].push(key);
        }
    }
}

fn extend_with_parents(set: &HashSet<TileKey>, lod_levels: u8) -> HashSet<TileKey> {
    let mut out = set.clone();
    for key in set.iter().copied() {
        let mut current = key;
        while let Some(parent) = parent_tile(current, lod_levels) {
            out.insert(parent);
            current = parent;
        }
    }
    out
}

fn required_parents_of_set(set: &HashSet<TileKey>, lod_levels: u8) -> HashSet<TileKey> {
    let mut required = HashSet::new();
    for key in set.iter().copied() {
        let mut current = key;
        while let Some(parent) = parent_tile(current, lod_levels) {
            required.insert(parent);
            current = parent;
        }
    }
    required
}

fn evict_subtree(set: &mut HashSet<TileKey>, root: TileKey, lod_levels: u8) -> usize {
    let mut to_remove = Vec::new();
    for key in set.iter().copied() {
        if is_descendant_or_self(key, root, lod_levels) {
            to_remove.push(key);
        }
    }
    for key in &to_remove {
        set.remove(key);
    }
    to_remove.len()
}

fn is_descendant_or_self(mut key: TileKey, ancestor: TileKey, lod_levels: u8) -> bool {
    if key == ancestor {
        return true;
    }
    while let Some(parent) = parent_tile(key, lod_levels) {
        if parent == ancestor {
            return true;
        }
        key = parent;
    }
    false
}

fn tile_priority(
    key: TileKey,
    activity_levels: &[ActivityLevel],
    tiles: &HashMap<TileKey, TileState>,
    frame: u64,
    lod_levels: u8,
) -> f32 {
    let activity = activity_at(activity_levels, key);
    let age = tiles
        .get(&key)
        .map(|state| frame.saturating_sub(state.last_active_frame) as f32)
        .unwrap_or(0.0);
    let recency = 1.0 / (1.0 + age);
    let lod_bias = (lod_levels.saturating_sub(1) as f32 - key.lod as f32) * 0.05;
    activity + recency * 0.2 + lod_bias
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

fn render_lod_pixels(
    level: &LodLevel,
    activity: &ActivityLevel,
    selected: &HashSet<TileKey>,
    camera: (f32, f32),
    scale: usize,
    max_dim: i32,
) -> Option<(u32, u32, Vec<u8>)> {
    if activity.tiles_x <= 0 || activity.tiles_z <= 0 {
        return None;
    }
    if activity.tiles_x > max_dim || activity.tiles_z > max_dim {
        return None;
    }
    let scale = scale.max(1);
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

    Some((width as u32, height as u32, pixels))
}

fn blit_pixels(
    dest: &mut [u8],
    dest_w: usize,
    dest_h: usize,
    src: Vec<u8>,
    src_w: usize,
    src_h: usize,
    offset_x: usize,
    offset_y: usize,
) {
    if offset_x >= dest_w || offset_y >= dest_h {
        return;
    }
    let copy_w = src_w.min(dest_w - offset_x);
    let copy_h = src_h.min(dest_h - offset_y);

    for y in 0..copy_h {
        let dest_row = (offset_y + y) * dest_w;
        let src_row = y * src_w;
        for x in 0..copy_w {
            let dest_idx = (dest_row + offset_x + x) * 3;
            let src_idx = (src_row + x) * 3;
            dest[dest_idx..dest_idx + 3].copy_from_slice(&src[src_idx..src_idx + 3]);
        }
    }
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
        ",mem_mb,new_tiles,evicted_tiles,budget_tiles,budget_evicted,budget_blocked,thrash_per_frame,thrash_ratio,pass_surface,pass_flux,pass_depth,pass_erosion,pass_upsample,pass_downsample\n",
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
            "image_composite" => cfg.image_composite = parse_bool(value)?,
            "thrash_window" => cfg.thrash_window = parse_u32(value)?,
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
