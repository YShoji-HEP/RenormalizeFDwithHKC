#![feature(f128)]
mod bessel;
mod bounce;
mod determinant;
mod heatkernel;
mod potential;
mod radial_wkb;
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
    let nu_max = 25;

    // let k = 0.4;
    // let nu_max = 50;

    // let z_list = [
    //     0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2,
    // ]
    // .map(|x| x * k);
    let z_list = [1.];

    // let calc_fd = z_list.len() == 1;
    // let calc_lam = z_list.len() == 1;
    // let calc_hke = z_list.len() > 1;

    let calc_fd = false;
    let calc_lam = false;
    let calc_hke = false;
    let calc_rwkb = true;

    let rho_ini = 1e-4;
    let step = 3e-4;

    let v = PhiFour::new(k);
    let mut bnc = Bounce::new(v, 4.);
    bnc.find_profile(1e-11, 80, rho_ini, step);
    bnc.rho_max = bnc.rho_max * 0.9;

    let lam =
        |_: usize, rho: f128, _: f128| [rho, rho.powi(3), rho.powi(5), rho.powi(7), rho.powi(9)];

    let hke = |nu: usize, rho: f128, z: f128| {
        let sqrt_z = z.sqrt();
        [
            rho * (rho * sqrt_z).besselik(nu),
            -rho.powi(2) / 2. / sqrt_z * (rho * sqrt_z).besselik_deriv(nu, 1),
            rho.powi(2) / 4. / sqrt_z.powi(3)
                * (rho * sqrt_z * (rho * sqrt_z).besselik_deriv(nu, 2)
                    - (rho * sqrt_z).besselik_deriv(nu, 1)),
            -rho.powi(2) / 8. / sqrt_z.powi(5)
                * ((rho * sqrt_z).powi(2) * (rho * sqrt_z).besselik_deriv(nu, 3)
                    - 3. * rho * sqrt_z * (rho * sqrt_z).besselik_deriv(nu, 2)
                    + 3. * (rho * sqrt_z).besselik_deriv(nu, 1)),
            rho.powi(2) / 16. / sqrt_z.powi(7)
                * ((rho * sqrt_z).powi(3) * (rho * sqrt_z).besselik_deriv(nu, 4)
                    - 6. * (rho * sqrt_z).powi(2) * (rho * sqrt_z).besselik_deriv(nu, 3)
                    + 15. * rho * sqrt_z * (rho * sqrt_z).besselik_deriv(nu, 2)
                    - 15. * (rho * sqrt_z).besselik_deriv(nu, 1)),
        ]
    };

    /////////////// Debug
    // bnc.ratio(20., rho_ini, step);panic!();
    // bnc.hk(0, &lam, 0., rho_ini, step);
    // let nu = 20;
    // bnc.hk(nu, &hke, mhat.powi(2), rho_ini, step);
    ///////////////

    let xi_approx = |nu: f128| {
        let d_nu = nu.powi(2);
        [
            arr1(&[d_nu * 1. / 2. / nu, 0., 0., 0., 0.]),
            arr1(&[d_nu * 1. / 2. / nu, d_nu * 1. / 4. / nu.powi(3), 0., 0., 0.]),
            arr1(&[
                d_nu * 1. / 2. / nu,
                d_nu * 1. / 4. / nu.powi(3) * (1. + 1. / nu.powi(2)),
                d_nu * 3. / 8. / nu.powi(5),
                0.,
                0.,
            ]),
            arr1(&[
                d_nu * 1. / 2. / nu,
                d_nu * 1. / 4. / nu.powi(3) * (1. + 1. / nu.powi(2) + 1. / nu.powi(4)),
                d_nu * 3. / 8. / nu.powi(5) * (1. + 5. / nu.powi(2)),
                d_nu * 15. / 16. / nu.powi(7),
                0.,
            ]),
            arr1(&[
                d_nu * 1. / 2. / nu,
                d_nu * 1. / 4. / nu.powi(3)
                    * (1. + 1. / nu.powi(2) + 1. / nu.powi(4) + 1. / nu.powi(6)),
                d_nu * 3. / 8. / nu.powi(5) * (1. + 5. / nu.powi(2) + 21. / nu.powi(4)),
                d_nu * 15. / 16. / nu.powi(7) * (1. + 14. / nu.powi(2)),
                d_nu * 105. / 32. / nu.powi(9),
            ]),
        ]
    };

    let res_lam = if calc_lam {
        bnc.hk(0, &lam, 0., rho_ini, step)
    } else {
        arr1(&[0.; 5])
    };
    for z in z_list {
        let mut handle = vec![];
        for nu in 1usize..nu_max + 1 {
            let mut bnc = bnc.clone();
            let res_lam = res_lam.clone();
            handle.push(thread::spawn(move || {
                let d_nu = nu.pow(2) as f128;

                let ratio = bnc.ratio(nu as f128, rho_ini, step);

                let dnu_lndet = d_nu * ratio[0].abs().ln();

                if calc_fd {
                    let fd_1 = dnu_lndet - d_nu * ratio[1];
                    let fd_2 = fd_1 - d_nu * (ratio[2] - ratio[1].powi(2) / 2.);
                    let fd_3 =
                        fd_2 - d_nu * (ratio[3] - ratio[1] * ratio[2] + ratio[1].powi(3) / 3.);

                    bbclient::post(
                        "lndet",
                        &format!("k:{}", k as f64),
                        [
                            nu as f64,
                            dnu_lndet.abs() as f64,
                            fd_1.abs() as f64,
                            fd_2.abs() as f64,
                            fd_3.abs() as f64,
                        ]
                        .into(),
                    )
                    .unwrap();
                }

                if calc_rwkb {
                    let rwkb = bnc.rwkb(nu, rho_ini, step);
                    let rwkb_2 = dnu_lndet + rwkb[0];
                    let rwkb_1 = rwkb_2 + rwkb[1];
                    let rwkb_0 = rwkb_1 + rwkb[2] + rwkb[3];
                    let rwkb_m1 = rwkb_0 + rwkb[4];
                    let rwkb_m2 = rwkb_m1 + rwkb[5];
                    let rwkb_m3 = rwkb_m2 + rwkb[6];
                    let rwkb_m4 = rwkb_m3 + rwkb[7];

                    bbclient::post(
                        "rwkb",
                        &format!("k:{}, z:{}", k as f64, z as f64),
                        [
                            nu as f64,
                            rwkb_2.abs() as f64,
                            rwkb_1.abs() as f64,
                            rwkb_0.abs() as f64,
                            rwkb_m1.abs() as f64,
                            rwkb_m2.abs() as f64,
                            rwkb_m3.abs() as f64,
                            rwkb_m4.abs() as f64,
                        ]
                        .into(),
                    )
                    .unwrap();
                }

                if calc_lam {
                    let lam_xi = xi_approx(nu as f128);
                    let lam_1 = dnu_lndet - (&res_lam * &lam_xi[0]).fold(0., |acc, x| acc + x);
                    let lam_2 = dnu_lndet - (&res_lam * &lam_xi[1]).fold(0., |acc, x| acc + x);
                    let lam_3 = dnu_lndet - (&res_lam * &lam_xi[2]).fold(0., |acc, x| acc + x);
                    let lam_4 = dnu_lndet - (&res_lam * &lam_xi[3]).fold(0., |acc, x| acc + x);
                    let lam_5 = dnu_lndet - (&res_lam * &lam_xi[4]).fold(0., |acc, x| acc + x);

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
                }

                if calc_hke {
                    let hke_temp = bnc.hk(nu, &hke, z, rho_ini, step);
                    let hke_1 = dnu_lndet - d_nu * hke_temp[0];
                    let hke_2 = hke_1 - d_nu * hke_temp[1];
                    let hke_3 = hke_2 - d_nu * hke_temp[2];
                    let hke_4 = hke_3 - d_nu * hke_temp[3];
                    let hke_5 = hke_4 - d_nu * hke_temp[4];

                    bbclient::post(
                        "hke",
                        &format!("k:{}, z:{}", k as f64, z as f64),
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
                }
            }));
        }
        for h in handle {
            h.join().unwrap();
        }
    }
}
