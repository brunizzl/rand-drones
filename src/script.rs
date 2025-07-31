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
                debug_assert!(eq_class.is_sorted());
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

    pub fn nr_unique_fst_paths(&self, nr_sites: usize) -> Option<usize> {
        let mut res = factorial(nr_sites)?;
        for class in &self.0 {
            res /= factorial(class.len())?;
        }
        Some(res)
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
        let mut sep = "";
        for path in self.iter() {
            write!(f, "{sep}{path:?}")?;
            sep = " ";
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
    best.sort_by_key(|x| x.0);
    best
}

/// computes the exact same result as [`compute_best_options_sequential`],
/// but utilizes multiple threads while doing so..
pub fn compute_best_options_parallel(
    site_probs: &[Prob],
    nr_drones: usize,
    nr_kept_paths: usize,
) -> Vec<(Prob, DronePaths)> {
    use rayon::prelude::*;

    let nr_sites = site_probs.len();
    let equiv = EquivSites::new(site_probs);
    // a single parallel job always keeps the path of either the first drone or the first two drones constant.
    // the first two drones are used to split jobs only if the first drone alone has too few different paths.
    let nr_start_drones = {
        let few_first_paths = equiv
            .nr_unique_fst_paths(nr_sites)
            .is_some_and(|nr| nr < 10);
        if few_first_paths { 2 } else { 1 }
    }
    .min(nr_drones);
    // all possible permutations of paths of the first one or two drones together in one looooong vector
    // (note: i was not able to get the parallel iterator to work with some lazy structure, thus we do this instead.)
    let start_drone_paths_flat = {
        let mut res = Vec::new();
        let mut curr = DronePaths::new(nr_start_drones, 0..nr_sites);
        res.extend_from_slice(&curr.flat);
        while curr.next_paths_permutation(&equiv).is_some() {
            res.extend_from_slice(&curr.flat);
        }
        res
    };
    let start_drones_paths = start_drone_paths_flat
        .chunks(nr_start_drones * nr_sites)
        .collect_vec();
    assert!(
        nr_start_drones != 1
            || Some(start_drones_paths.len()) == equiv.nr_unique_fst_paths(nr_sites)
    );
    let mut best = start_drones_paths
        .par_iter()
        .map(|start_drones_path| {
            let mut paths = {
                // if there is only a single drone, then d1 == d2
                let d1 = 0..nr_sites;
                let d2 = (start_drones_path.len() - nr_sites)..;
                assert!(start_drones_path[d1.clone()] <= start_drones_path[d2.clone()]);
                let mut paths = DronePaths::new(nr_drones, start_drones_path[d2].iter().copied());
                paths[0].copy_from_slice(&start_drones_path[d1]);
                paths
            };
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
                // if the change in paths happend in one of the paths we split the
                // parallel jobs over, then this job is done.
                if paths
                    .next_paths_permutation(&equiv)
                    .is_none_or(|i| i <= (nr_start_drones - 1) * nr_sites)
                {
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
    best.sort_by_key(|x| x.0);
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

    Prob::new((nr_successfull as f64) / (nr_samples.max(1) as f64))
}

#[allow(dead_code)]
fn run_fixed_instance(nr_drones: usize, site_probs: &[Prob]) {
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

fn best_results(solution: &[(Prob, DronePaths)]) -> &[(Prob, DronePaths)] {
    assert!(solution.is_sorted_by_key(|x| x.0));
    let best_prob = solution.last().unwrap().0;
    let cutoff = best_prob & Prob::new(0.99999);
    let start = solution.iter().position(|(p, _)| *p >= cutoff).unwrap();
    &solution[start..]
}

/// returns the indices of all sites that can be removed 
/// in order for an optimal solution of the given problem to be the same as an optimal solution
/// for the problem with a site removed, modulo the removed site.
/// e.g. if `[0, 3, 1, 2]` is a (path of an) optimal solution of the full problem 
/// and if `[0, 1, 2]` is a (path of an) optimal solution to the problem with site `3` taken away, 
/// then the original solution can be build by inserting site `3` somewhere in the previous solution.
/// hence: in the example, the returned values would include `3`.
#[allow(dead_code)]
fn removable_sites(nr_drones: usize, site_probs: &[Prob]) -> Vec<usize> {
    let nr_sites = site_probs.len();
    let full_solution = compute_best_options_parallel(&site_probs, nr_drones, 100);
    let bests = best_results(&full_solution);

    let mut res = Vec::new();
    for i in 0..nr_sites {
        let mut smaller_probs = Vec::from(site_probs);
        // note: replacing one site with a site of success probability 1 is the same as removing this site:
        // the new site can be inserted everywhere in a drone path and not change the outcome.
        smaller_probs[i] = Prob::ALWAYS;
        let nr_reduced_kept = 10 * bests.len() * nr_sites.pow(nr_drones as u32);
        let reduced_solution = compute_best_options_parallel(&smaller_probs, nr_drones, nr_reduced_kept);
        let reduced_bests = best_results(&reduced_solution);
        // ensure that the function will not fail because we failed to keep enough solutions.
        assert!(reduced_bests.len() < reduced_solution.len());

        let mut i_is_removable = false;
        for (_, best) in bests {
            i_is_removable |= reduced_bests.iter().any(|(_, rb)| rb == best);
        }
        if i_is_removable {
            res.push(i);
        }
    }
    res
}

pub fn test_random_sites_for_removability() {
    let nr_sites = 5;
    let nr_drones = 3;
    for _ in 0..100 {
        let site_probs = {
            let mut res = Vec::new();
            for _ in 0..nr_sites {
                let site_prob = Prob::new(rand::random());
                res.push(site_prob);
            }
            res.sort();
            res
        };
        let removable = removable_sites(nr_drones, &site_probs);
        for &p in &site_probs {
            print!("{p} ");
        }
        let min_diff = {
            let mut pairs = vec![Prob::NEVER; nr_sites + 2];
            pairs[1..(1 + nr_sites)].copy_from_slice(&site_probs);
            pairs[1 + nr_sites] = Prob::ALWAYS;
            let int_diff = |(a, b): (&Prob, &Prob)| ((b.value() - a.value()) * 1e10) as u64;
            pairs.iter().tuple_windows().map(int_diff).min().unwrap() as f64 * 1e-10
        };
        print!("(diff >= {min_diff:>.6})");
        println!(" -> {removable:?}");
    }
}

pub fn main() {
    //let site_probs = [
    //    Prob::new(0.2988236),
    //    Prob::new(0.2988236),
    //    Prob::new(0.2988236),
    //    Prob::new(0.2988236),
    //    Prob::new(0.2988236),
    //    Prob::new(0.2988236),
    //];
    //let site_probs = [Prob::new(0.002); 6];
    //let nr_drones = 3;

    //run_fixed_instance(nr_drones, &site_probs);

    //let removable = removable_sites(nr_drones, &site_probs);
    //println!("{removable:?}");

    test_random_sites_for_removability();
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
        let best = best_results(&results);
        // TODO: all permutations of a single path should be `factorial(5)`.
        // why do we only see half that?
        assert_eq!(best.len(), factorial(site_probs.len()).unwrap() / 2);
        for (_, paths) in best {
            let snd_path_rev = paths[1].iter().cloned().rev().collect_vec();
            assert_eq!(&paths[0], &snd_path_rev[..]);
        }
    }

    #[test]
    fn parallel_eq_sequential_1() {
        let site_probs = [
            Prob::new(0.25),
            Prob::new(0.25),
            Prob::new(0.25),
            Prob::new(0.6),
            Prob::new(0.8),
        ];
        assert_eq!(
            EquivSites::new(&site_probs).nr_unique_fst_paths(site_probs.len()),
            // note: this is >= 10, thus parallel splits threads only depending on the fst drone's path
            Some(5 * 4 * 3 * 2 * 1 / (2 * 3))
        );

        let nr_drones = 3;
        let nr_kept_paths = 200;
        let res_par = compute_best_options_parallel(&site_probs, nr_drones, nr_kept_paths);
        let res_seq = compute_best_options_sequential(&site_probs, nr_drones, nr_kept_paths);
        assert_eq!(res_par, res_seq);
    }

    #[test]
    fn parallel_eq_sequential_2() {
        let site_probs = [
            Prob::new(0.25),
            Prob::new(0.25),
            Prob::new(0.5),
            Prob::new(0.5),
        ];
        assert_eq!(
            EquivSites::new(&site_probs).nr_unique_fst_paths(site_probs.len()),
            // note: this is < 10, thus parallel splits threads depending also on snd drone's path
            Some(4 * 3 * 2 * 1 / 2 / 2)
        );

        let nr_drones = 3;
        let nr_kept_paths = 200;
        let res_par = compute_best_options_parallel(&site_probs, nr_drones, nr_kept_paths);
        let res_seq = compute_best_options_sequential(&site_probs, nr_drones, nr_kept_paths);
        assert_eq!(res_par, res_seq);
    }
}
