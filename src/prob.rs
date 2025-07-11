/// short for _Propability_, thus the held data is assumed to be in the range `0.0..=1.0`.
/// All operations on multiple propabilites assume they are independent from each other.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prob(f64);

impl Prob {
    pub const ALWAYS: Self = Self(1.0);
    pub const NEVER: Self = Self(0.0);

    pub fn value(self) -> f64 {
        self.0
    }

    pub fn new(val: f64) -> Self {
        debug_assert!((0.0..=1.0).contains(&val));
        Self(val)
    }

    /// if two events have two independent given propabilities,
    /// this computes the propability that none occur.
    pub fn nand(self, other: Self) -> Self {
        !self & !other
    }

    pub fn all(ps: impl IntoIterator<Item = Self>) -> Self {
        ps.into_iter().fold(Self::ALWAYS, |a, b| a & b)
    }

    pub fn none(ps: impl IntoIterator<Item = Self>) -> Self {
        Self::all(ps.into_iter().map(|x| !x))
    }

    pub fn any(ps: impl IntoIterator<Item = Self>) -> Self {
        !Self::none(ps)
    }
}

impl std::fmt::Display for Prob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.5}", self.0)
    }
}

impl std::ops::BitAnd for Prob {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl std::ops::BitOr for Prob {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        !Self::nand(self, rhs)
    }
}

impl std::ops::BitOrAssign for Prob {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs
    }
}

impl std::ops::BitAndAssign for Prob {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs
    }
}

impl std::ops::Not for Prob {
    type Output = Self;
    fn not(self) -> Self::Output {
        Self(1.0 - self.0)
    }
}

impl std::ops::BitXor for Prob {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        (self | rhs) & !(self & rhs)
    }
}

/// values can never be nan -> no worries
impl std::cmp::Eq for Prob {}

impl std::cmp::PartialOrd for Prob {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for Prob {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn operators() {
        assert_eq!(Prob::new(0.5) & Prob::new(0.5), Prob::new(0.25));
        assert_eq!(Prob::new(0.25) | Prob::new(0.25), Prob::new(0.4375));
        assert_eq!(!Prob::new(0.125), Prob::new(0.875));
    }

    #[test]
    fn iters() {
        let ps = [
            Prob::new(0.5),
            Prob::new(0.64),
            Prob::new(0.12),
            Prob::new(0.333),
        ];
        assert_eq!(ps[0] & ps[1] & ps[2] & ps[3], Prob::all(ps.into_iter()));
        assert_eq!(ps[0] | ps[1] | ps[2] | ps[3], Prob::any(ps.into_iter()));
    }
}
