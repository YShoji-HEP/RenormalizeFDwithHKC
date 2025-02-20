#![feature(f128)]
mod bessel;
mod bounce;
mod determinant;
mod heatkernel;
mod potential;
mod tools;

use bessel::BesselIK;
use bounce::Bounce;
use bulletin_board_client as bbclient;
use ndarray::arr1;
use potential::Potential;
use std::thread;

#[derive(Clone)]
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
    let _buf = dbgbb::Buffer::on();

    let k = 0.2;
    let v = PhiFour::new(k);

    let rho_ini = 1e-4;
    let step = 3e-4;

    let mut bnc = Bounce::new(v, 4.);
    bnc.find_profile(1e-11, 80, rho_ini, step);
    bnc.rho_max = bnc.rho_max * 0.9;

    let lam = |_: f128, rho: f128| [rho, rho.powi(3), rho.powi(5), rho.powi(7), rho.powi(9)];

    // bnc.ratio(20., rho_ini, step);
    // bnc.hk(0., &lam, 0., rho_ini, step);

    let xi_approx = |nu: f128| {
        let d_nu = nu.powi(2);
        arr1(&[
            d_nu * 1. / 2. / nu,
            d_nu * 1. / 4. / nu.powi(3)
                * (1. + 1. / nu.powi(2) + 1. / nu.powi(4) + 1. / nu.powi(6)),
            d_nu * 3. / 8. / nu.powi(5) * (1. + 5. / nu.powi(2) + 21. / nu.powi(4)),
            d_nu * 15. / 16. / nu.powi(7) * (1. + 14. / nu.powi(2)),
            d_nu * 105. / 32. / nu.powi(9),
        ])
    };

    let mhat = k.sqrt()*0.5;
    let res_lam = bnc.hk(0., &lam, 0., rho_ini, step);

    let mut handle = vec![];
    for nu in 0..22 {
        let mut bnc = bnc.clone();
        let res_lam = res_lam.clone();
        handle.push(thread::spawn(move || {
            let hke = |_: f128, rho: f128| {
                [
                    rho * (rho * mhat).besselik(nu),
                    -rho.powi(2) / 2. / mhat * (rho * mhat).besselik_deriv(nu, 1),
                    rho.powi(2) / 4. / mhat.powi(3)
                        * (rho * mhat * (rho * mhat).besselik_deriv(nu, 2)
                            - (rho * mhat).besselik_deriv(nu, 1)),
                    -rho.powi(2) / 8. / mhat.powi(5)
                        * ((rho * mhat).powi(2) * (rho * mhat).besselik_deriv(nu, 3)
                            - 3. * rho * mhat * (rho * mhat).besselik_deriv(nu, 2)
                            + 3. * (rho * mhat).besselik_deriv(nu, 1)),
                    rho.powi(2) / 16. / mhat.powi(7)
                        * ((rho * mhat).powi(3) * (rho * mhat).besselik_deriv(nu, 4)
                            - 6. * (rho * mhat).powi(2) * (rho * mhat).besselik_deriv(nu, 3)
                            + 15. * rho * mhat * (rho * mhat).besselik_deriv(nu, 2)
                            - 15. * (rho * mhat).besselik_deriv(nu, 1)),
                ]
            };
            let ratio = bnc.ratio(nu as f128, rho_ini, step);

            let d_nu = nu.pow(2) as f128;

            let lndet = ratio[0].abs().ln();

            let fd_1 = d_nu * (lndet - ratio[1]);
            let fd_2 = fd_1 - d_nu * (ratio[2] + ratio[1].powi(2) / 2.);
            let fd_3 = fd_2 - d_nu * (ratio[3] + ratio[1] * ratio[2] - ratio[1].powi(3) / 3.);

            let lam_temp = &res_lam * xi_approx(nu as f128);
            let lam_1 = d_nu * lndet - lam_temp[0];
            let lam_2 = lam_1 - lam_temp[1];
            let lam_3 = lam_2 - lam_temp[2];
            let lam_4 = lam_3 - lam_temp[3];
            let lam_5 = lam_4 - lam_temp[4];

            let hke_temp = bnc.hk(nu as f128, &hke, mhat.powi(2), rho_ini, step);
            let hke_1 = d_nu * (lndet - hke_temp[0]);
            let hke_2 = hke_1 - d_nu * hke_temp[1];
            let hke_3 = hke_2 - d_nu * hke_temp[2];
            let hke_4 = hke_3 - d_nu * hke_temp[3];
            let hke_5 = hke_4 - d_nu * hke_temp[4];

            bbclient::post(
                "lndet",
                &format!("k:{}", k as f64),
                [
                    nu as f64,
                    lndet.abs() as f64,
                    fd_1.abs() as f64,
                    fd_2.abs() as f64,
                    fd_3.abs() as f64,
                ]
                .into(),
            )
            .unwrap();

            bbclient::post(
                "lam",
                &format!("k:{}", k as f64),
                [
                    nu as f64,
                    lam_1.abs() as f64,
                    lam_2.abs() as f64,
                    lam_3.abs() as f64,
                    lam_4.abs() as f64,
                    lam_5.abs() as f64,
                ]
                .into(),
            )
            .unwrap();

            bbclient::post(
                "hke",
                &format!("k:{}", k as f64),
                [
                    nu as f64,
                    hke_1.abs() as f64,
                    hke_2.abs() as f64,
                    hke_3.abs() as f64,
                    hke_4.abs() as f64,
                    hke_5.abs() as f64,
                ]
                .into(),
            )
            .unwrap();
        }));
    }
    for h in handle {
        h.join().unwrap();
    }
}
