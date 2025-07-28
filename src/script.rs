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
/// note 1: although not assumed in ths function, it is of no utility to not visit every site with every drone.
/// note 2: the site success probabilities are not indepentent.
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
/// note: this function has exponential complexity.
pub fn prob_mission_succeedes<'a>(
    site_probs: &[Prob],
    drone_paths: impl Iterator<Item = &'a [usize]> + Clone,
) -> Prob {
    let nr_sites = site_probs.len();
    debug_assert!(drone_paths_valid(nr_sites, drone_paths.clone()));
    debug_assert!(drone_paths_non_repeating(drone_paths.clone()));

    const NOT_VISITED: usize = usize::MAX;
    let mut visited_sites = vec![NOT_VISITED; nr_sites];

    /// if `i_drone` many drones have flown so far
    /// and successfully visited every site `j` where `visited_sites[j] != NOT_VISITED`,
    /// what is the probability that the mission succedes overall?
    fn prob_remaining_mission_succeedes<'b>(
        i_drone: usize,
        visited_sites: &mut [usize],
        site_probs: &[Prob],
        mut remaining_drone_paths: impl Iterator<Item = &'b [usize]> + Clone,
    ) -> Prob {
        if !visited_sites.contains(&NOT_VISITED) {
            return Prob::ALWAYS;
        }
        let Some(drone_i_path) = remaining_drone_paths.next() else {
            return Prob::NEVER;
        };
        let remaining_success = |visited_sites: &mut [usize]| {
            prob_remaining_mission_succeedes(
                i_drone + 1,
                visited_sites,
                site_probs,
                remaining_drone_paths.clone(),
            )
        };
        // this loop looks at disjoint events.
        // either drone_i dies at exactly the first visited site, or the second or ... or drone_i survives.
        // because two of these events are impossible to happen simultaniously,
        // we can simply add the success chances of every case.
        let mut drone_i_still_alive = Prob::ALWAYS;
        let mut mission_success = Prob::NEVER;
        for &drone_i_site in drone_i_path {
            let drone_i_dies_now = drone_i_still_alive & !site_probs[drone_i_site];
            mission_success += drone_i_dies_now & remaining_success(visited_sites);

            drone_i_still_alive &= site_probs[drone_i_site];
            if visited_sites[drone_i_site] == NOT_VISITED {
                visited_sites[drone_i_site] = i_drone;
            }
        }
        mission_success += drone_i_still_alive & remaining_success(visited_sites);

        for site in visited_sites {
            if *site == i_drone {
                *site = NOT_VISITED;
            }
        }

        mission_success
    }

    prob_remaining_mission_succeedes(0, &mut visited_sites, site_probs, drone_paths)
}

/// turns `nums` into the next lexicographic larger permutation.
/// if no such permutation exists, `nums` is turned in the smallest permutation and `false` is returned.
pub fn next_permutation<T: Ord>(nums: &mut [T]) -> bool {
    let Some(left) = nums.windows(2).rposition(|w| w[0] < w[1]) else {
        nums.reverse();
        return false;
    };
    let right_off = left + 1;
    let gt_left = |n| n > &nums[left];
    debug_assert!(gt_left(&nums[right_off]));
    // can always unwrap, because we know right_off fulfills condition.
    let right = nums[right_off..].iter().rposition(gt_left).unwrap();
    nums.swap(left, right_off + right);
    nums[right_off..].reverse();
    true
}

/// can only store those paths, where each drone visits each site exactly once.
/// (any optimal set of paths satisfies this condition)
#[derive(Debug, Clone)]
pub struct DronePaths {
    /// stores all paths of length `self.nr_sites` continuously in memory
    flat: Vec<usize>,
    /// stores chunk size of [`Self::flat`]
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
/// the success probability of all possible paths (where each drone visits each site exactly once)
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

    all_paths_with_probs.sort_by_key(|x| x.0);
    all_paths_with_probs
}

fn simulate_success<'a>(
    nr_samples: usize,
    site_probs: &[Prob],
    drone_paths: impl Iterator<Item = &'a [usize]> + Clone + Sync,
) -> Prob {
    let nr_sites = site_probs.len();
    use rayon::prelude::*;
    let nr_successfull: usize = vec![(); nr_samples]
        .par_iter()
        .map(|_| {
            use smallvec::{SmallVec, smallvec};
            let mut visited_vec: SmallVec<[bool; 16]> = smallvec![false; nr_sites];
            let visited = &mut visited_vec[..];
            for path in drone_paths.clone() {
                for &site in path {
                    if site_probs[site].sample() {
                        visited[site] = true;
                    } else {
                        break;
                    }
                }
            }
            let success = visited.iter().all(|&x| x);
            if success { 1usize } else { 0usize }
        })
        .sum();

    Prob::new((nr_successfull as f64) / (nr_samples as f64))
}

pub fn main() {
    let site_probs = [
        Prob::new(0.41),
        Prob::new(0.5),
        Prob::new(0.62),
        Prob::new(0.83),
        Prob::new(0.87),
        //Prob::new(0.92),
        //Prob::new(0.96),
    ];
    let nr_drones = 3;

    let res = compute_all_options(&site_probs, nr_drones);
    println!("computed {} results.", res.len());

    let print_res_slice = |rs: &[(Prob, DronePaths)]| {
        for (prob, paths) in rs {
            let estimate = simulate_success(10_000_000, &site_probs, paths.iter());
            println!("{prob} ({estimate}) {paths}");
        }
    };
    if res.len() > 1000 {
        print_res_slice(&res[..50]);
        println!("...");
        print_res_slice(&res[(res.len() - 50)..]);
    } else {
        print_res_slice(&res);
    }
}

#[cfg(test)]
mod text {
    use itertools::izip;

    use super::*;

    #[test]
    fn next_permutation_works_1() {
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

    #[test]
    fn next_permutation_works_2() {
        let mut sample = [0, 0, 1, 0];
        assert!(next_permutation(&mut sample));
        assert_eq!(sample, [0, 1, 0, 0]);
        assert!(next_permutation(&mut sample));
        assert_eq!(sample, [1, 0, 0, 0]);
        assert!(!next_permutation(&mut sample));
        assert_eq!(sample, [0, 0, 0, 1]);
    }

    #[test]
    fn number_permutations_correct() {
        let mut order = [0, 1, 2, 3, 4, 5];
        let mut nr_permutations = 1;
        while next_permutation(&mut order) {
            nr_permutations += 1;
        }
        assert_eq!(nr_permutations, 6 * 5 * 4 * 3 * 2 * 1);
    }

    #[test]
    fn number_drone_paths_correct() {
        let mut drone_paths = DronePaths::new(3, 5);
        let mut nr_paths = 1;
        while drone_paths.next_paths_permutation() {
            nr_paths += 1;
        }

        /// returns the number of distinct mutlisets with [`cardinality`] many elements
        /// chosen from a set with [`universe_size`] many elements.
        fn multiset_count(universe_size: usize, cardinality: usize) -> Option<usize> {
            fn binomial_coefficient(n: usize, k: usize) -> Option<usize> {
                if k == 0 {
                    return Some(0);
                }
                let mut res: usize = 1;
                for i in 1..=k {
                    res = res.checked_mul(n + 1 - i)?;
                    debug_assert_eq!(res % i, 0);
                    res /= i;
                }
                Some(res)
            }
            debug_assert_eq!(binomial_coefficient(10, 3), Some(120));
            debug_assert_eq!(binomial_coefficient(20, 10), Some(184756));
            debug_assert_eq!(binomial_coefficient(8, 6), Some(28));

            binomial_coefficient(universe_size + cardinality - 1, cardinality)
        }

        let universe_size = 5 * 4 * 3 * 2 * 1;
        let cardinality = 3;
        let nr_multisets = multiset_count(universe_size, cardinality);

        assert_eq!(Some(nr_paths), nr_multisets);
    }

    #[test]
    fn singe_site_mission() {
        let p = Prob::new(0.25);
        let site_probs = [p];
        let drone_paths = [[0], [0], [0], [0]];
        let success = prob_mission_succeedes(&site_probs, drone_paths.iter().map(|x| &x[..]));

        assert_eq!(success, p | p | p | p);
    }

    #[test]
    fn same_order_drones() {
        let site_probs = [Prob::new(0.25), Prob::new(0.5), Prob::new(0.75)];
        let nr_drones = 4;
        let drone_paths_1 = std::iter::repeat_n(&[0usize, 1, 2] as &[_], nr_drones);
        let drone_paths_2 = std::iter::repeat_n(&[2usize, 1, 0] as &[_], nr_drones);
        let drone_paths_3 = std::iter::repeat_n(&[1usize, 2, 0] as &[_], nr_drones);

        let success = Prob::any(std::iter::repeat_n(Prob::all(site_probs), nr_drones));
        let success_1 = prob_mission_succeedes(&site_probs, drone_paths_1);
        let success_2 = prob_mission_succeedes(&site_probs, drone_paths_2);
        let success_3 = prob_mission_succeedes(&site_probs, drone_paths_3);
        assert_eq!(success, success_1);
        assert_eq!(success, success_2);
        assert_eq!(success, success_3);
    }

    #[test]
    fn two_drones_optimal() {
        let site_probs = [
            Prob::new(0.3),
            Prob::new(0.4),
            Prob::new(0.6),
            Prob::new(0.8),
            Prob::new(0.9),
        ];
        let results = compute_all_options(&site_probs, 2);
        let best_prob = results.last().unwrap().0;
        let best_results = results
            .into_iter()
            .filter_map(|(p, res)| (p >= best_prob & Prob::new(0.9999999)).then_some(res))
            .collect_vec();
        // TODO: all permutations of a single path should be `factorial(5)`.
        // why do we only see half that?
        assert_eq!(best_results.len(), (5 * 4 * 3 * 2 * 1) / 2);
        for paths in best_results {
            for (site_a, site_b) in izip!(paths[0].iter(), paths[1].iter().rev()) {
                assert_eq!(site_a, site_b);
            }
        }
    }
}
