#![feature(f128)]
mod bounce;
mod determinant;
mod heatkernel;
mod potential;
mod tools;
mod bessel;

use bounce::Bounce;
use ndarray::arr1;
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
    let rho_ini = 1e-4;
    let step = 3e-4;
    bnc.find_profile(1e-10, 80, rho_ini, step);
    bnc.rho_max = bnc.rho_max * 0.9;
    let lam = |_: f128, rho: f128| [rho, rho.powi(3), rho.powi(5), rho.powi(7), rho.powi(9)];
    let xi_approx = |nu: f128| {
        arr1(&[
            1. / 2. / nu,
            1. / 4. / nu.powi(3) * (1. + 1. / nu.powi(2) + 1. / nu.powi(4) + 1. / nu.powi(6)),
            3. / 8. / nu.powi(5) * (1. + 5. / nu.powi(2) + 21. / nu.powi(4)),
            15. / 16. / nu.powi(7) * (1. + 14. / nu.powi(2)),
            105. / 32. / nu.powi(9),
        ])
    };

    let res_lam = bnc.hk(0., &lam, rho_ini, step, false);
    bnc.ratio(20., rho_ini, step, true);
    // bnc.hk(20., drho, &i_nu, true);
    for nu in 0..30 {
        let ratio = bnc.ratio(nu as f128, rho_ini, step, false);
        dbgbb::dbgbb!(((ratio[0].ln()
            - (&res_lam * xi_approx(nu as f128)).fold(0., |acc, x| acc + x))
            as f64)
            .rename("subtracted"));
        dbgbb::dbgbb!(ratio.map(|&x| x as f64).rename("ratio"));
        dbgbb::dbgbb!((&res_lam * xi_approx(nu as f128))
            .map(|&x| x as f64)
            .rename("lam"));
        // dbgbb::dbgbb!(bnc
        //     .hk(i as f128, drho, &i_nu, false)
        //     .map(|x| x as f64)
        //     .rename("hkc"));
        dbg!(nu);
    }
}
