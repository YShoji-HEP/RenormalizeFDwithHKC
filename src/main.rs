#![feature(f128)]
mod bounce;
mod determinant;
mod heatkernel;
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
    // fn val(&self, phi: f128) -> f128 {
    //     1. / 4. * phi.powi(4) - (self.k + 1.) / 3. * phi.powi(3) + self.k / 2. * phi.powi(2)
    // }
    fn first_deriv(&self, phi: f128) -> f128 {
        phi.powi(3) - (self.k + 1.) * phi.powi(2) + self.k * phi
    }
    fn second_deriv(&self, phi: f128) -> f128 {
        3. * phi.powi(2) - 2. * (self.k + 1.) * phi + self.k
    }
    fn third_deriv(&self, phi: f128) -> f128 {
        6. * phi - 2. * (self.k + 1.)
    }
    fn forth_deriv(&self, _: f128) -> f128 {
        6.
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
    fn second_deriv_fv(&self) -> f128 {
        self.k
    }
}

fn main() {
    let v = PhiFour::new(0.2);
    let mut bnc = Bounce::new(v, 4.);
    let drho = 1e-4;
    bnc.find_profile(drho, 0.00001, 50);
    let i_nu = |nu: f128, rho: f128| {
        [
            rho / 2. / nu,
            rho.powi(3) / 4. / nu.powi(3)
                * (1. + 1. / nu.powi(2) + 1. / nu.powi(4) + 1. / nu.powi(6)),
            rho.powi(5) * 3. / 8. / nu.powi(5) * (1. + 5. / nu.powi(2) + 21. / nu.powi(4)),
            rho.powi(7) * 15. / 16. / nu.powi(7) * (1. + 14. / nu.powi(2)),
            rho.powi(9) * 105. / 32. / nu.powi(9),
        ]
    };
    // for i in 0..30 {
    //     dbgbb::dbgbb!(bnc
    //         .ratio(i as f128, drho, false)
    //         .map(|x| x as f64)
    //         .rename("ratio"));
    //     dbgbb::dbgbb!(bnc
    //         .hk(i as f128, drho, &i_nu, false)
    //         .map(|x| x as f64)
    //         .rename("hkc"));
    // }
    bnc.hk(20., drho, &i_nu, true);
    // bnc.ratio(19., drho);
    // dbgbb::dbgbb!(
    //     bnc.rho.map(|&x| x as f64).rename("rho"),
    //     bnc.phi.map(|&x| x as f64).rename("phi"),
    //     bnc.psi0_nu.map(|&x| x as f64).rename("psi0"),
    //     (&bnc.psi_nu / &bnc.psi0_nu)
    //         .map(|&x| x as f64)
    //         .rename("psi"),
    //     bnc.phi.map(|&x| x as f64).rename("phi"),
    //     (&bnc.psi1_nu / &bnc.psi0_nu)
    //         .map(|&x| x as f64)
    //         .rename("psi1"),
    //     bnc.phi.map(|&x| x as f64).rename("phi"),
    //     (&bnc.psi2_nu / &bnc.psi0_nu)
    //         .map(|&x| x as f64)
    //         .rename("psi2"),
    //     bnc.err.map(|&x| x as f64).rename("err")
    // );
}
