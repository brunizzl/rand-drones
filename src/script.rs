use itertools::Itertools;

use super::Prob;

/// the indices of sites visited by each drone must be valid.
fn drone_paths_valid<'a>(
    nr_sites: usize,
    mut drone_paths: impl Iterator<Item = &'a [usize]>,
) -> bool {
    drone_paths.all(|path| path.iter().all(|&s| s < nr_sites))
}

/// each drone may visit each site at most once
fn drone_paths_non_repeating<'a>(drone_paths: impl Iterator<Item = &'a [usize]>) -> bool {
    let mut visited = Vec::new();
    for drone_path in drone_paths {
        visited.clear();
        visited.extend_from_slice(drone_path);
        visited.sort();
        if visited.iter().tuple_windows().any(|(a, b)| a == b) {
            return false;
        }
    }
    true
}

/// `site` indexes into `site_probs`, so does every item in every item of `drone_paths`.
/// the number of flown drones is `drone_paths.count()`.
/// note: although not assumed in ths function, it is of no utility to not visit every site with every drone.
pub fn prob_site_succeedes<'a>(
    site: usize,
    site_probs: &[Prob],
    drone_paths: impl Iterator<Item = &'a [usize]> + Clone,
) -> Prob {
    let nr_sites = site_probs.len();
    debug_assert!(site < nr_sites);
    debug_assert!(drone_paths_valid(nr_sites, drone_paths.clone()));

    let relevant_paths_parts = drone_paths.filter_map(|path| {
        path.iter()
            .position(|&s| s == site)
            .map(|i| &path[..(i + 1)])
    });
    let chance_per_drone =
        relevant_paths_parts.map(|path| Prob::all(path.iter().map(|&s| site_probs[s])));
    Prob::any(chance_per_drone)
}

/// the mission succeedes, if every site is successfully visited by at least one drone.
/// note: except for a worse runtime complexity,
/// this function should be indistinguisable from [`prob_mission_succeedes`].
pub fn prob_mission_succeedes_<'a>(
    site_probs: &[Prob],
    drone_paths: impl Iterator<Item = &'a [usize]> + Clone,
) -> Prob {
    let nr_sites = site_probs.len();
    let individual_site_successes =
        (0..nr_sites).map(|site| prob_site_succeedes(site, site_probs, drone_paths.clone()));
    Prob::all(individual_site_successes)
}

/// same as [`prob_mission_succeedes_`], except assumes each single drone to visit each site at most once.
/// note: unlike [`prob_mission_succeedes_`], this implementation allocates.
pub fn prob_mission_succeedes<'a>(
    site_probs: &[Prob],
    drone_paths: impl Iterator<Item = &'a [usize]> + Clone,
) -> Prob {
    let nr_sites = site_probs.len();
    debug_assert!(drone_paths_valid(nr_sites, drone_paths.clone()));
    debug_assert!(drone_paths_non_repeating(drone_paths.clone()));

    let mut site_success_props = vec![Prob::NEVER; nr_sites];
    for drone_path in drone_paths {
        let mut survival_prop = Prob::ALWAYS;
        for &site in drone_path {
            survival_prop &= site_probs[site];
            site_success_props[site] |= survival_prop;
        }
    }
    Prob::all(site_success_props)
}

/// turns `nums` into the next lexicographic larger permutation.
/// if no such permutation exists, `nums` is turned in the smallest permutation and `false` is returned.
/// stolen from https://codereview.stackexchange.com/questions/259168/rust-implementation-of-next-permutation.
pub fn next_permutation<T: Ord>(nums: &mut [T]) -> bool {
    use std::cmp::Ordering;
    let Some(last_ascending) = nums.windows(2).rposition(|w| w[0] < w[1]) else {
        nums.reverse();
        return false;
    };

    let swap_with = nums[(last_ascending + 1)..]
        .binary_search_by(|n| T::cmp(&nums[last_ascending], n).then(Ordering::Less))
        .unwrap_err(); // cannot fail because the binary search will never succeed
    nums.swap(last_ascending, last_ascending + swap_with);
    nums[last_ascending + 1..].reverse();
    true
}

/// can only store those paths, where each drone visits each site exactly once.
/// (any optimal set of paths satisfies this condition)
#[derive(Debug, Clone)]
pub struct DronePaths {
    /// stores all paths of length `self.nr_sites` continuously in memory
    flat: Vec<usize>,
    nr_sites: usize,
}

impl std::fmt::Display for DronePaths {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for path in self.iter() {
            write!(f, "{path:?} ")?;
        }
        Ok(())
    }
}

impl std::ops::Index<usize> for DronePaths {
    type Output = [usize];
    fn index(&self, index: usize) -> &Self::Output {
        let start = index * self.nr_sites;
        let stop = start + self.nr_sites;
        &self.flat[start..stop]
    }
}

impl std::ops::IndexMut<usize> for DronePaths {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let start = index * self.nr_sites;
        let stop = start + self.nr_sites;
        &mut self.flat[start..stop]
    }
}

impl DronePaths {
    pub fn new(nr_drones: usize, nr_sites: usize) -> Self {
        let flat = (0..nr_drones).flat_map(|_| 0..nr_sites).collect_vec();
        Self { flat, nr_sites }
    }

    pub fn nr_drones(&self) -> usize {
        debug_assert_eq!(self.flat.len() % self.nr_sites, 0);
        self.flat.len() / self.nr_sites
    }

    pub fn iter<'a>(&'a self) -> std::slice::Chunks<'a, usize> {
        debug_assert_eq!(self.flat.len() % self.nr_sites, 0);
        self.flat.chunks(self.nr_sites)
    }

    pub fn iter_mut<'a>(&'a mut self) -> std::slice::ChunksMut<'a, usize> {
        debug_assert_eq!(self.flat.len() % self.nr_sites, 0);
        self.flat.chunks_mut(self.nr_sites)
    }

    /// reorders drone paths, so that the next lexicographic larger combination of paths
    /// is produced. if no such ordering exists, `false` is returned.
    pub fn next_paths_permutation(&mut self) -> bool {
        let mut path_start = self.flat.len();
        debug_assert!(path_start % self.nr_sites == 0);
        while path_start != 0 {
            let path_end = path_start;
            path_start -= self.nr_sites;
            if next_permutation(&mut self.flat[path_start..path_end]) {
                let (fixed, reset) = self.flat.split_at_mut(path_end);
                let changed_path = &fixed[path_start..path_end];
                for path_to_reset in reset.chunks_mut(self.nr_sites) {
                    path_to_reset.copy_from_slice(changed_path);
                }
                return true;
            }
        }

        false
    }
}

/// for the given set of sites and the given number of drones,
/// the success propability of all possible paths (where each drone visits each site exactly once)
/// is evaluated and returned.
pub fn compute_all_options(site_probs: &[Prob], nr_drones: usize) -> Vec<(Prob, DronePaths)> {
    let nr_sites = site_probs.len();
    let mut paths = DronePaths::new(nr_drones, nr_sites);
    let mut all_paths_with_probs = Vec::new();
    loop {
        let prob = prob_mission_succeedes(site_probs, paths.iter());
        all_paths_with_probs.push((prob, paths.clone()));

        if !paths.next_paths_permutation() {
            break;
        }
    }

    all_paths_with_probs.sort_by(|a, b| a.0.cmp(&b.0));
    all_paths_with_probs
}

pub fn main() {
    let site_probs = [Prob::new(0.4), Prob::new(0.5), Prob::new(0.7)];
    let nr_drones = 6;
    let res = compute_all_options(&site_probs, nr_drones);
    for (prob, paths) in res {
        println!("{prob} {paths}");
    }
}

#[cfg(test)]
mod text {
    use super::*;

    #[test]
    fn both_prop_mission_succeedes_impls_eq() {
        let sites = [
            Prob::new(0.8),
            Prob::new(0.3),
            Prob::new(0.1),
            Prob::new(0.5),
        ];
        let paths = [
            [0, 1, 2, 3],
            [3, 1, 2, 0],
            [1, 2, 0, 3],
            [3, 0, 2, 1],
            [1, 0, 3, 2],
        ];
        let p1 = prob_mission_succeedes_(&sites, paths.iter().map(|path| &path[..]));
        let p2 = prob_mission_succeedes(&sites, paths.iter().map(|path| &path[..]));
        assert_eq!(p1, p2);
    }

    #[test]
    fn next_permutation_works() {
        let mut sample = [1, 2, 3, 4];
        assert!(next_permutation(&mut sample));
        assert_eq!(sample, [1, 2, 4, 3]);
        assert!(next_permutation(&mut sample));
        assert_eq!(sample, [1, 3, 2, 4]);
        assert!(next_permutation(&mut sample));
        assert_eq!(sample, [1, 3, 4, 2]);
        assert!(next_permutation(&mut sample));
        assert_eq!(sample, [1, 4, 2, 3]);

        sample = [4, 3, 2, 1];
        assert!(!next_permutation(&mut sample));
        assert_eq!(sample, [1, 2, 3, 4]);
    }
}
