#![feature(f128)]
mod bounce;
mod potential;
mod tools;

use bounce::Bounce;
use potential::Potential;

struct PhiFour {
    k: f128,
}

impl PhiFour {
    fn new(k: f128) -> Self {
        Self { k }
    }
}
impl Potential for PhiFour {
    fn val(&self, phi: f128) -> f128 {
        1. / 4. * phi.powi(4) - (self.k + 1.) / 3. * phi.powi(3) + self.k / 2. * phi.powi(2)
    }
    fn first_deriv(&self, phi: f128) -> f128 {
        phi.powi(3) - (self.k + 1.) * phi.powi(2) + self.k * phi
    }
    fn second_deriv(&self, phi: f128) -> f128 {
        3. * phi.powi(2) - 2. * (self.k + 1.) * phi + self.k
    }
    fn phi_fv(&self) -> f128 {
        0.
    }
    fn phi_tv(&self) -> f128 {
        1.
    }
    fn phi_top(&self) -> f128 {
        self.k
    }
}

fn main() {
    let v = PhiFour::new(0.2);
    let mut bnc = Bounce::new(v, 4.);
    bnc.find_profile(1e-4, 30);
    dbgbb::dbgbb!(
        bnc.rho.map(|&x| x as f64),
        bnc.phi.map(|&x| x as f64),
        bnc.err.map(|&x| x as f64)
    );
}
