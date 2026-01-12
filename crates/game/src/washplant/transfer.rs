/// Axis-aligned bounding box for spatial queries
#[derive(Clone, Debug)]
pub struct AABB {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl AABB {
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Check if point is inside AABB (inclusive)
    pub fn contains(&self, point: [f32; 3]) -> bool {
        point[0] >= self.min[0] && point[0] <= self.max[0] &&
        point[1] >= self.min[1] && point[1] <= self.max[1] &&
        point[2] >= self.min[2] && point[2] <= self.max[2]
    }

    /// Get center point of AABB
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) / 2.0,
            (self.min[1] + self.max[1]) / 2.0,
            (self.min[2] + self.max[2]) / 2.0,
        ]
    }

    /// Get size (width, height, depth) of AABB
    pub fn size(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }
}

/// Particle in transit between zones with arrival time
#[derive(Clone, Debug)]
pub struct TransitParticle {
    pub particle_index: usize,
    pub arrival_time: f32,
}

impl TransitParticle {
    pub fn new(particle_index: usize, arrival_time: f32) -> Self {
        Self { particle_index, arrival_time }
    }
}

/// Transfer zone for capturing and moving particles
pub struct TransferZone {
    pub capture_region: AABB,
    pub transit_duration: f32,
    transit_queue: Vec<TransitParticle>,
}

impl TransferZone {
    pub fn new(capture_region: AABB, transit_duration: f32) -> Self {
        Self {
            capture_region,
            transit_duration,
            transit_queue: Vec::new(),
        }
    }

    /// Check if particle position should be captured
    pub fn should_capture(&self, position: [f32; 3]) -> bool {
        self.capture_region.contains(position)
    }

    /// Enqueue particle for transfer
    pub fn enqueue(&mut self, particle_index: usize, current_time: f32) {
        let arrival_time = current_time + self.transit_duration;
        self.transit_queue.push(TransitParticle::new(particle_index, arrival_time));
    }

    /// Process transit queue, return particles that have arrived
    pub fn tick(&mut self, current_time: f32) -> Vec<usize> {
        let mut arrived = Vec::new();
        self.transit_queue.retain(|p| {
            if p.arrival_time <= current_time {
                arrived.push(p.particle_index);
                false
            } else {
                true
            }
        });
        arrived
    }

    /// Get count of particles in transit
    pub fn in_transit(&self) -> usize {
        self.transit_queue.len()
    }

    /// Clear all transit queue
    pub fn clear(&mut self) {
        self.transit_queue.clear();
    }
}

/// Statistics for transfer zone activity
#[derive(Clone, Debug, Default)]
pub struct TransferStats {
    pub total_captured: usize,
    pub total_delivered: usize,
    pub current_in_transit: usize,
}

impl TransferStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_capture(&mut self) {
        self.total_captured += 1;
        self.current_in_transit += 1;
    }

    pub fn record_delivery(&mut self, count: usize) {
        self.total_delivered += count;
        self.current_in_transit = self.current_in_transit.saturating_sub(count);
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_contains_inside() {
        let aabb = AABB::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        assert!(aabb.contains([5.0, 5.0, 5.0]));
    }

    #[test]
    fn test_aabb_contains_boundary() {
        let aabb = AABB::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        assert!(aabb.contains([0.0, 0.0, 0.0]));
        assert!(aabb.contains([10.0, 10.0, 10.0]));
        assert!(aabb.contains([5.0, 0.0, 10.0]));
    }

    #[test]
    fn test_aabb_contains_outside() {
        let aabb = AABB::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        assert!(!aabb.contains([-0.1, 5.0, 5.0]));
        assert!(!aabb.contains([5.0, 10.1, 5.0]));
        assert!(!aabb.contains([5.0, 5.0, -1.0]));
    }

    #[test]
    fn test_aabb_center() {
        let aabb = AABB::new([0.0, 0.0, 0.0], [10.0, 20.0, 30.0]);
        let center = aabb.center();
        assert_eq!(center, [5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_aabb_size() {
        let aabb = AABB::new([1.0, 2.0, 3.0], [11.0, 22.0, 33.0]);
        let size = aabb.size();
        assert_eq!(size, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_transfer_zone_capture_enqueue_tick() {
        let capture_region = AABB::new([0.0, 0.0, 0.0], [5.0, 5.0, 5.0]);
        let mut zone = TransferZone::new(capture_region, 2.0);

        assert!(zone.should_capture([2.5, 2.5, 2.5]));
        assert!(!zone.should_capture([10.0, 10.0, 10.0]));

        zone.enqueue(42, 1.0);
        assert_eq!(zone.in_transit(), 1);

        let arrived = zone.tick(2.5);
        assert_eq!(arrived.len(), 0);
        assert_eq!(zone.in_transit(), 1);

        let arrived = zone.tick(3.0);
        assert_eq!(arrived, vec![42]);
        assert_eq!(zone.in_transit(), 0);
    }

    #[test]
    fn test_transfer_zone_clear() {
        let capture_region = AABB::new([0.0, 0.0, 0.0], [5.0, 5.0, 5.0]);
        let mut zone = TransferZone::new(capture_region, 2.0);

        zone.enqueue(1, 1.0);
        zone.enqueue(2, 1.5);
        assert_eq!(zone.in_transit(), 2);

        zone.clear();
        assert_eq!(zone.in_transit(), 0);
    }

    #[test]
    fn test_transfer_stats() {
        let mut stats = TransferStats::new();
        assert_eq!(stats.total_captured, 0);
        assert_eq!(stats.total_delivered, 0);
        assert_eq!(stats.current_in_transit, 0);

        stats.record_capture();
        assert_eq!(stats.total_captured, 1);
        assert_eq!(stats.current_in_transit, 1);

        stats.record_capture();
        assert_eq!(stats.total_captured, 2);
        assert_eq!(stats.current_in_transit, 2);

        stats.record_delivery(1);
        assert_eq!(stats.total_delivered, 1);
        assert_eq!(stats.current_in_transit, 1);

        stats.reset();
        assert_eq!(stats.total_captured, 0);
        assert_eq!(stats.total_delivered, 0);
        assert_eq!(stats.current_in_transit, 0);
    }

    #[test]
    fn test_transfer_stats_saturating_sub() {
        let mut stats = TransferStats::new();
        stats.record_delivery(10); // Should not underflow
        assert_eq!(stats.current_in_transit, 0);
    }
}
