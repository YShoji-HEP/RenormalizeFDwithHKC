mod bounce;
mod potential;
mod tools;

use bounce::Bounce;
use potential::Potential;

struct PhiFour {
    k: f64,
}

impl PhiFour {
    fn new(k: f64) -> Self {
        Self { k }
    }
}
impl Potential for PhiFour {
    fn val(&self, phi: &f64) -> f64 {
        1. / 4. * phi.powi(4) - (self.k + 1.) / 3. * phi.powi(3) + self.k / 2. * phi.powi(2)
    }
    fn first_deriv(&self, phi: &f64) -> f64 {
        phi.powi(3) - (self.k + 1.) * phi.powi(2) + self.k * phi
    }
    fn second_deriv(&self, phi: &f64) -> f64 {
        3. * phi.powi(2) - 2. * (self.k + 1.) * phi + self.k
    }
    fn phi_fv(&self) -> f64 {
        0.
    }
    fn phi_tv(&self) -> f64 {
        1.
    }
    fn phi_top(&self) -> f64 {
        self.k
    }
}
fn main() {
    let v = PhiFour::new(0.2);
    let bnc = Bounce::new(v);
}
