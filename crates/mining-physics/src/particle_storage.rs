//! Derived allocation policy. No allocation count is a physical clogging law.
use mining_sim::sluice::INITIAL_GRAIN_STORAGE;
// Conservative request planning, not a claim about measured peak process RSS.
const PARTICLE_PLANNING_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
const PLANNED_BYTES_PER_PARTICLE: u64 = 1024;

#[cfg(test)]
pub(crate) fn grain_capacity(
    water: usize,
    needed: usize,
    minimum: usize,
    max_particle_vector_bytes: u64,
) -> Result<usize, String> {
    grain_capacity_with_extra(water, needed, minimum, max_particle_vector_bytes, 0, 0)
}

pub(crate) fn grain_capacity_with_extra(
    water: usize,
    needed: usize,
    minimum: usize,
    max_particle_vector_bytes: u64,
    fixed_bytes: u64,
    extra_bytes_per_particle: u64,
) -> Result<usize, String> {
    let grains = needed
        .max(minimum)
        .max(INITIAL_GRAIN_STORAGE)
        .checked_next_power_of_two()
        .ok_or("Particle storage count overflow")?;
    let count = water
        .checked_add(grains)
        .ok_or("Particle storage count overflow")?;
    let count = u64::try_from(count).map_err(|_| "Particle storage count overflow")?;
    let planned = count
        .saturating_mul(PLANNED_BYTES_PER_PARTICLE.saturating_add(extra_bytes_per_particle))
        .saturating_add(fixed_bytes);
    if planned > PARTICLE_PLANNING_BUDGET_BYTES
        || count.saturating_mul(16) > max_particle_vector_bytes
    {
        return Err("Particle storage request exceeds the planning/adapter budget; simulation must pause with its material retained".into());
    }
    Ok(grains)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocation_grows_geometrically_and_checks_limits_before_allocating() {
        let binding = 128 * 1024 * 1024;
        assert_eq!(grain_capacity(300_000, 4096, 0, binding).unwrap(), 4096);
        assert_eq!(grain_capacity(300_000, 4097, 0, binding).unwrap(), 8192);
        assert_eq!(grain_capacity(400_000, 4096, 0, binding).unwrap(), 4096);
        assert_eq!(grain_capacity(300_000, 80, 8192, binding).unwrap(), 8192);
        assert!(grain_capacity(300_000, usize::MAX, 0, binding).is_err());
        assert!(grain_capacity(300_000, 4097, 0, 128).is_err());
        assert!(grain_capacity(300_000, 524_288, 0, binding).is_err());
        assert!(
            grain_capacity_with_extra(400_000, 8192, 0, binding, 128 * 1024 * 1024, 8).is_err()
        );
        assert_eq!(
            grain_capacity_with_extra(300_000, 8192, 0, binding, 32 * 1024 * 1024, 8).unwrap(),
            8192
        );
    }
}
