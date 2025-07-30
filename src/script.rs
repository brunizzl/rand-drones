use itertools::{Itertools, izip};

use super::Prob;

/// the indices of sites visited by each drone must be valid.
fn drone_paths_valid<'a>(
    nr_sites: usize,
    mut drone_paths: impl Iterator<Item = &'a [usize]>,
) -> bool {
    drone_paths.all(|path| path.iter().all(|&s| s < nr_sites))
}

/// the mission succeedes, if every site is successfully visited by at least one drone.
/// this function implements the main computation part of the script.
pub fn prob_mission_succeedes<'a>(
    site_probs: &[Prob],
    drone_paths: impl Iterator<Item = &'a [usize]> + Clone,
) -> Prob {
    use std::collections::HashMap;
    let nr_sites = site_probs.len();
    debug_assert!(drone_paths_valid(nr_sites, drone_paths.clone()));
    assert!(
        nr_sites <= std::mem::size_of::<usize>() * 8,
        "memoisation relies on fixed bound on number of sites for efficient memory use."
    );

    const NOT_VISITED: usize = usize::MAX;
    let mut visited_sites = vec![NOT_VISITED; nr_sites];
    /// pack nr_sites many bools which determine which site was visited
    /// (thus it is no longer differentiated, which drone visited which site)
    fn pack_visited_sites(visited_sites: &[usize]) -> usize {
        let mut res = 0;
        for &s in visited_sites {
            // `true` becomes `1`, `false` becomes `0`
            const _: () = assert!(true as usize == 1 && false as usize == 0);
            res |= (s != NOT_VISITED) as usize;
            res <<= 1;
        }
        res
    }
    let mut memory = HashMap::new();

    /// if `i_drone` many drones have flown so far
    /// and successfully visited every site `j` where `visited_sites[j] != NOT_VISITED`,
    /// what is the probability that the mission succedes overall?
    fn prob_remaining_mission_succeedes<'b>(
        i_drone: usize,
        memory: &mut HashMap<(usize, usize), Prob>,
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
        let memoised_state = (pack_visited_sites(visited_sites), i_drone);
        if let Some(val) = memory.get(&memoised_state) {
            return *val;
        }

        // this loop looks at disjoint events.
        // either drone_i dies at exactly the first visited site, or the second or ... or drone_i survives.
        // because two of these events are impossible to happen simultaniously,
        // we can simply add the success chances of every case.
        let mut drone_i_still_alive = Prob::ALWAYS;
        let mut mission_success = Prob::NEVER;
        // this variable could be recomputed every loop iteration,
        // but if the currently visited site was already visited by an earlier drone, the value stays the same.
        // because the exponential explosion happens in the recursion calls when this is recomputed,
        // we try to not do it more than required.
        let mut finish_remaining = prob_remaining_mission_succeedes(
            i_drone + 1,
            memory,
            visited_sites,
            site_probs,
            remaining_drone_paths.clone(),
        );
        for &drone_i_site in drone_i_path {
            let drone_i_dies_now = drone_i_still_alive & !site_probs[drone_i_site];
            mission_success += drone_i_dies_now & finish_remaining;

            drone_i_still_alive &= site_probs[drone_i_site];
            if visited_sites[drone_i_site] == NOT_VISITED {
                visited_sites[drone_i_site] = i_drone;
                // `visited_sites` was changed, so we must update the variable depending on it.
                finish_remaining = prob_remaining_mission_succeedes(
                    i_drone + 1,
                    memory,
                    visited_sites,
                    site_probs,
                    remaining_drone_paths.clone(),
                );
            }
        }
        let drone_i_never_dies = drone_i_still_alive;
        mission_success += drone_i_never_dies & finish_remaining;

        for site in visited_sites {
            if *site == i_drone {
                *site = NOT_VISITED;
            }
        }

        memory.insert(memoised_state, mission_success);
        mission_success
    }

    prob_remaining_mission_succeedes(0, &mut memory, &mut visited_sites, site_probs, drone_paths)
}

#[allow(dead_code)]
fn factorial(mut n: usize) -> Option<usize> {
    let mut res = 1usize;
    while n > 0 {
        res = res.checked_mul(n)?;
        n -= 1;
    }
    Some(res)
}

/// returns the number of distinct mutlisets with [`cardinality`] many elements
/// chosen from a set with [`universe_size`] many elements.
#[allow(dead_code)]
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

/// turns `nums` into the next lexicographic larger permutation.
/// if no such permutation exists, `nums` is turned in the smallest permutation and `false` is returned.
pub fn next_permutation<T: PartialOrd>(nums: &mut [T]) -> bool {
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

/// if two or more sites have the same success probability,
/// all sites with the same probability make up one entry of [`Self::0`].
#[derive(Debug, Clone)]
pub struct EquivSites(Vec<Vec<usize>>);

impl EquivSites {
    pub fn new(site_probs: &[Prob]) -> Self {
        let mut with_indices = izip!(0.., site_probs.iter().copied()).collect_vec();
        with_indices.sort_by_key(|x| x.1);
        let mut res = Vec::new();
        while let Some((i1, p1)) = with_indices.pop() {
            let mut eq_class = vec![i1];
            while let Some(&(i2, p2)) = with_indices.last()
                && p1 == p2
            {
                with_indices.pop();
                eq_class.push(i2);
            }
            if eq_class.len() > 1 {
                eq_class.reverse();
                debug_assert!(eq_class.iter().is_sorted());
                res.push(eq_class);
            }
        }
        Self(res)
    }

    /// returns true iff every eqivalence class of `self` is ordered in `path`.
    pub fn eqs_ordered(&self, path: &[usize]) -> bool {
        for class in &self.0 {
            let mut i = 0;
            for &elem in class {
                while path[i] != elem {
                    i += 1;
                    if i == path.len() {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// can only store those paths, where each drone visits each site exactly once.
/// (any optimal set of paths satisfies this condition)
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// returns `nr_drones` copies of `init_path`
    pub fn new(nr_drones: usize, init_path: impl Iterator<Item = usize> + Clone) -> Self {
        let flat = (0..nr_drones).flat_map(|_| init_path.clone()).collect_vec();
        let nr_sites = init_path.count();
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

    /// reorders drone paths, so that the next lexicographic larger combination of paths is produced.
    /// if such an ordering exists, the index of the start of the first changed path segment is returned,
    /// otherwise `None`.
    /// note: in this program, the interesting cases of the output are `None`, `Some(0)` and the rest.
    pub fn next_paths_permutation(&mut self, equiv: &EquivSites) -> Option<usize> {
        let mut path_start = self.flat.len();
        debug_assert!(path_start % self.nr_sites == 0);
        while path_start != 0 {
            let path_end = path_start;
            path_start -= self.nr_sites;
            let found_next = loop {
                let path = &mut self.flat[path_start..path_end];
                let could_permute = next_permutation(path);
                if path_start != 0 || !could_permute || equiv.eqs_ordered(path) {
                    break could_permute;
                }
            };
            if found_next {
                let (fixed, reset) = self.flat.split_at_mut(path_end);
                let changed_path = &fixed[path_start..path_end];
                for path_to_reset in reset.chunks_mut(self.nr_sites) {
                    path_to_reset.copy_from_slice(changed_path);
                }
                return Some(path_start);
            }
        }

        None
    }
}

/// for the given set of sites and the given number of drones,
/// the success probability of all possible paths (where each drone visits each site exactly once)
/// is evaluated and returned.
pub fn compute_all_options(site_probs: &[Prob], nr_drones: usize) -> Vec<(Prob, DronePaths)> {
    compute_best_options_parallel(site_probs, nr_drones, usize::MAX >> 4)
}

/// same as [`compute_all_options`], except only keeps the specified number of best options.
/// note: this sequential implementation is mainly kept around to verify the parallel implmentation.
pub fn compute_best_options_sequential(
    site_probs: &[Prob],
    nr_drones: usize,
    nr_kept_paths: usize,
) -> Vec<(Prob, DronePaths)> {
    let start_time = std::time::Instant::now();

    let nr_sites = site_probs.len();
    let eqiv = EquivSites::new(site_probs);
    let mut paths = DronePaths::new(nr_drones, 0..nr_sites);
    let mut cutoff_prob = Prob::NEVER;
    let mut best = Vec::new();

    loop {
        let prob = prob_mission_succeedes(site_probs, paths.iter());
        if prob > cutoff_prob {
            best.push((prob, paths.clone()));
            if best.len() / 2 >= nr_kept_paths {
                best.sort_by_key(|x| !x.0);
                best.truncate(nr_kept_paths);
                cutoff_prob = best.last().unwrap().0;
            }
        }

        if paths.next_paths_permutation(&eqiv).is_none() {
            break;
        }
    }

    best.sort_by_key(|x| !x.0);
    best.truncate(nr_kept_paths);
    best.reverse();

    let end_time = std::time::Instant::now();
    println!(
        "sequential implementation took {:?}.",
        end_time - start_time
    );

    best
}

/// computes the exact same result as [`compute_best_options_sequential`],
/// but utilizes all the machines cores while doing so.
/// note: because the jobs are split to handle cases for different paths of the first drone,
/// more threads are only used if the sites form more than one equivalent class.
pub fn compute_best_options_parallel(
    site_probs: &[Prob],
    nr_drones: usize,
    nr_kept_paths: usize,
) -> Vec<(Prob, DronePaths)> {
    use rayon::prelude::*;
    let start_time = std::time::Instant::now();

    let nr_sites = site_probs.len();
    let equiv = EquivSites::new(site_probs);
    // all possible permutations of the first drone path packed together in one looooong vector
    // (note: i was not able to get the parallel iterator to work with some lazy structure, thus we do this instead.)
    let fst_drone_paths_flat = {
        let mut res = Vec::new();
        let mut curr = DronePaths::new(1, 0..nr_sites);
        res.extend_from_slice(&curr[0]);
        while curr.next_paths_permutation(&equiv).is_some() {
            res.extend_from_slice(&curr[0]);
        }
        res
    };
    let fst_drone_paths = fst_drone_paths_flat.chunks(nr_sites).collect_vec();
    let mut best = fst_drone_paths
        .par_iter()
        .map(|fst_drone_path| {
            let mut paths = DronePaths::new(nr_drones, fst_drone_path.iter().copied());
            let mut cutoff_prob = Prob::NEVER;
            let mut best = Vec::new();
            loop {
                let prob = prob_mission_succeedes(site_probs, paths.iter());
                if prob > cutoff_prob {
                    best.push((prob, paths.clone()));
                    if best.len() / 2 >= nr_kept_paths {
                        best.sort_by_key(|x| !x.0);
                        best.truncate(nr_kept_paths);
                        cutoff_prob = best.last().unwrap().0;
                    }
                }
                if paths.next_paths_permutation(&equiv).is_none_or(|i| i == 0) {
                    break;
                }
            }
            best
        })
        .reduce(Vec::new, |mut a, b| {
            a.extend_from_slice(&b);
            a.sort_by_key(|x| !x.0);
            a.truncate(nr_kept_paths);
            a
        });
    best.reverse();

    let end_time = std::time::Instant::now();
    println!("parallel implementation took {:?}.", end_time - start_time);

    best
}

/// this function tries to compute the same as [`prob_mission_succeedes`], but by experiment.
/// if the other approach is done correctly, it only suffers from floating point errors.
/// we can't hope to get this precise in reasonable time with the simulation approach.
/// this function therefore exists mainly to validate the implementation of [`prob_mission_succeedes`].
fn simulate_success_prob<'a>(
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
    //let site_probs = [
    //    Prob::new(0.41),
    //    Prob::new(0.5),
    //    //Prob::new(0.551),
    //    Prob::new(0.62),
    //    Prob::new(0.83),
    //    Prob::new(0.87),
    //    //Prob::new(0.92),
    //    //Prob::new(0.96),
    //];
    let site_probs = [Prob::new(0.5); 5];
    let nr_drones = 5;
    let nr_kept = 100;
    let nr_simulations = 1_000;

    let res = compute_best_options_parallel(&site_probs, nr_drones, nr_kept);
    //let res = compute_all_options(&site_probs, nr_drones);
    println!("show best {} results.", res.len());
    for (prob, paths) in res {
        let estimate = simulate_success_prob(nr_simulations, &site_probs, paths.iter());
        println!("{prob} (ca. {estimate})   {paths}");
    }
}

#[cfg(test)]
mod text {
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
        assert_eq!(nr_permutations, factorial(order.len()).unwrap());
    }

    #[test]
    fn number_drone_paths_correct() {
        let mut drone_paths = DronePaths::new(3, 0..5);
        let equiv = EquivSites(Vec::new());
        let mut nr_paths = 1;
        while drone_paths.next_paths_permutation(&equiv).is_some() {
            nr_paths += 1;
        }

        assert_eq!(Some(nr_paths), multiset_count(factorial(5).unwrap(), 3));
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
        let drone_paths_1 = DronePaths::new(nr_drones, [0usize, 1, 2].into_iter());
        let drone_paths_2 = DronePaths::new(nr_drones, [2usize, 1, 0].into_iter());
        let drone_paths_3 = DronePaths::new(nr_drones, [1usize, 2, 0].into_iter());

        let success = Prob::any(std::iter::repeat_n(Prob::all(site_probs), nr_drones));
        let success_1 = prob_mission_succeedes(&site_probs, drone_paths_1.iter());
        let success_2 = prob_mission_succeedes(&site_probs, drone_paths_2.iter());
        let success_3 = prob_mission_succeedes(&site_probs, drone_paths_3.iter());
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
        let cutoff = best_prob & Prob::new(0.9999999);
        let best_results = results
            .into_iter()
            .filter_map(|(p, res)| (p >= cutoff).then_some(res))
            .collect_vec();
        // TODO: all permutations of a single path should be `factorial(5)`.
        // why do we only see half that?
        assert_eq!(best_results.len(), factorial(site_probs.len()).unwrap() / 2);
        for paths in best_results {
            let snd_path_rev = paths[1].iter().cloned().rev().collect_vec();
            assert_eq!(&paths[0], &snd_path_rev[..]);
        }
    }

    #[test]
    fn parallel_eq_sequential() {
        let site_probs = [
            Prob::new(0.3),
            Prob::new(0.4),
            Prob::new(0.6),
            Prob::new(0.8),
        ];
        let nr_drones = 3;
        let nr_kept_paths = 200;
        let res_par = compute_best_options_parallel(&site_probs, nr_drones, nr_kept_paths);
        let res_seq = compute_best_options_sequential(&site_probs, nr_drones, nr_kept_paths);
        assert_eq!(res_par, res_seq);
    }
}
