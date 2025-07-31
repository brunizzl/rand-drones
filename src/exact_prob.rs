
use fraction::{One, Zero, ToPrimitive};

//type BU = fraction::DynaInt<u64, fraction::BigUint>;
type BU = fraction::BigUint;
type Ratio = fraction::Ratio<BU>;


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExactProb(Ratio);

impl ExactProb {

    pub fn always() -> Self {
        Self(Ratio::one())
    }

    pub fn never() -> Self {
        Self(Ratio::zero())
    }

    pub fn approximate(&self) -> super::Prob {
        super::Prob::new(self.0.to_f64().unwrap())
    }

    pub fn new(val: f64) -> Self {
        debug_assert!((0.0..=1.0).contains(&val));
        const DEN: u64 = 1 << 32;
        let num = (val * (DEN as f64)) as u64;
        Self(Ratio::new(num.into(), DEN.into()))
    }

    pub fn all(ps: impl IntoIterator<Item = Self>) -> Self {
        ps.into_iter().fold(Self::always(), |a, b| { a & b } )
    }

    pub fn none(ps: impl IntoIterator<Item = Self>) -> Self {
        Self::all(ps.into_iter().map(|x| !x))
    }

    pub fn any(ps: impl IntoIterator<Item = Self>) -> Self {
        !Self::none(ps)
    }
}

impl std::fmt::Display for ExactProb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.approximate())
    }
}

impl std::ops::BitOr for &ExactProb {
    type Output = ExactProb;
    fn bitor(self, rhs: Self) -> Self::Output {
        let x = &self.0;
        let y = &rhs.0;
        ExactProb(x + y - x * y)
    }
}

impl std::ops::BitOr for ExactProb {
    type Output = ExactProb;
    fn bitor(self, rhs: Self) -> Self::Output {
        let x = self.0;
        let y = rhs.0;
        ExactProb(x.clone() + &y - x * y)
    }
}

impl std::ops::BitOr<&ExactProb> for ExactProb {
    type Output = ExactProb;
    fn bitor(self, rhs: &Self) -> Self::Output {
        let x = self.0;
        let y = rhs.0.clone();
        ExactProb(x.clone() + &y - x * y)
    }
}

impl std::ops::Not for ExactProb {
    type Output = ExactProb;
    fn not(self) -> Self::Output {
        let (num, den) = self.0.into_raw();
        ExactProb(Ratio::new(den.clone() - num, den))
    }
}

impl std::ops::Not for &ExactProb {
    type Output = ExactProb;
    fn not(self) -> Self::Output {
        !self.clone()
    }
}

impl std::ops::AddAssign<&ExactProb> for ExactProb {
    fn add_assign(&mut self, rhs: &Self) {
        self.0 += &rhs.0;
    }
}

impl std::ops::AddAssign for ExactProb {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl std::ops::BitAndAssign<&ExactProb> for ExactProb {
    fn bitand_assign(&mut self, rhs: &ExactProb) {
        self.0 *= &rhs.0;
    }
}

impl std::ops::BitAnd for &ExactProb {
    type Output = ExactProb;
    fn bitand(self, rhs: Self) -> Self::Output {
        ExactProb(&self.0 * &rhs.0)
    }
}

impl std::ops::BitAnd<&ExactProb> for ExactProb {
    type Output = ExactProb;
    fn bitand(mut self, rhs: &ExactProb) -> ExactProb {
        self &= rhs;
        self
    }
}

impl std::ops::BitAnd for ExactProb {
    type Output = ExactProb;
    fn bitand(mut self, rhs: ExactProb) -> ExactProb {
        self &= &rhs;
        self
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn operators() {
        assert_eq!(ExactProb::new(0.5) & ExactProb::new(0.5), ExactProb::new(0.25));
        assert_eq!(ExactProb::new(0.25) | ExactProb::new(0.25), ExactProb::new(0.4375));
        assert_eq!(!ExactProb::new(0.125), ExactProb::new(0.875));
    }

    #[test]
    fn iters() {
        let ps = [
            ExactProb::new(0.5),
            ExactProb::new(0.64),
            ExactProb::new(0.12),
            ExactProb::new(0.333),
        ];
        assert_eq!(&ps[0] & &ps[1] & &ps[2] & &ps[3], ExactProb::all(ps.iter().cloned()));
        assert_eq!(&ps[0] | &ps[1] | &ps[2] | &ps[3], ExactProb::any(ps.iter().cloned()));
    }
}
