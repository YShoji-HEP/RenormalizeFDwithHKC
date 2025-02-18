use crate::potential::*;
use crate::tools::*;
use ndarray::{arr1, Array1};
use ndarray_stats::QuantileExt;

const ABS_TOL: f64 = 1e-5;
const REL_TOL: f64 = 1e-5;

#[derive(Debug, PartialEq)]
pub enum ShootingResult {
    WrongDirection,
    OverShoot,
    UnderShoot,
    Success,
}

pub struct Bounce<T: Potential> {
    pub v: T,
    pub rho: Array1<f64>,
    pub phi_0: f64,
    pub phi: Array1<f64>,
    pub phi_deriv: Array1<f64>,
}

impl<T: Potential> Bounce<T> {
    pub fn new(v: T) -> Self {
        Self {
            v,
            rho: Default::default(),
            phi_0: Default::default(),
            phi: Default::default(),
            phi_deriv: Default::default(),
        }
    }
}

impl<T: Potential> Bounce<T> {
    pub fn find_profile(&mut self, max_iter: usize) {
        let mut phi_0_range = self.v.phi_top()..self.v.phi_tv();
        let mut i = 0;
        while i < max_iter {
            self.phi_0 = (phi_0_range.start + phi_0_range.end) / 2.;
            match self.shoot(self.phi_0, false) {
                ShootingResult::OverShoot => {
                    phi_0_range.end = self.phi_0;
                    i += 1;
                }
                ShootingResult::UnderShoot => {
                    phi_0_range.start = self.phi_0;
                    i += 1;
                }
                ShootingResult::Success => {
                    break;
                }
                ShootingResult::WrongDirection => {
                    phi_0_range.end = self.phi_0;
                }
            }
        }
        self.shoot(self.phi_0, true);
    }
    fn eom(&self, phi: f64, dphi: f64) -> f64 {
        1.
    }
    pub fn shoot(&mut self, phi_ini: f64, save: bool) -> ShootingResult {
        let mut drho = 1e-5;
        let mut rho = drho;
        let mut y = arr1(&[phi_ini, 0., rho, 1.]);
        let dydrho = |_: f64, fld: &Array1<f64>| {
            let phi = fld[0];
            let dphi = fld[1];
            let ddphi = self.eom(phi, dphi);
            arr1(&[dphi, ddphi])
        };
        let mut res_rho = vec![rho];
        let mut res_phi = vec![y[0]];
        let mut res_dphi = vec![y[1]];
        let mut dphi_max = std::f64::NEG_INFINITY;
        let res = loop {
            let (dy, err) = stepper::dp45(rho, &y, drho, &dydrho);

            let err = if err.sum().is_normal() {
                *(err.map(|x| x.abs()) / (REL_TOL * y.map(|x| x.abs()) + ABS_TOL))
                    .max()
                    .unwrap()
            } else {
                2.
            };
            if err > 1. || y[2] + dy[2] < 0. {
                drho *= 0.9;
                if drho < 1e-12 {
                    panic!();
                }
            } else {
                y += &dy;
                rho += drho;

                let phi = y[0];
                let dphi = y[1];

                if save {
                    res_rho.push(rho);
                    res_phi.push(phi);
                    res_dphi.push(dphi);
                }

                if dphi_max < dphi.abs() {
                    dphi_max = dphi.abs();
                }
                if phi > self.v.phi_tv() {
                    break ShootingResult::WrongDirection;
                }
                if (phi - self.v.phi_fv()).abs() < (self.v.phi_top() - self.v.phi_fv()).abs() {
                    if (phi - self.v.phi_fv()).abs()
                        < 0.001 * (self.v.phi_top() - self.v.phi_fv()).abs()
                        && dphi.abs() < 0.001 * dphi_max
                    {
                        break ShootingResult::Success;
                    }
                    if dphi.abs() < 0.001 * dphi_max {
                        break ShootingResult::Success;
                    }
                    if dphi > 0. {
                        break ShootingResult::UnderShoot;
                    }
                    if phi < self.v.phi_fv() {
                        break ShootingResult::OverShoot;
                    }
                }
                drho *= 1.02;
                drho = drho.min(1e-2);
            }
        };
        if save {
            self.rho = res_rho.into();
            self.phi = res_phi.into();
            self.phi_deriv = res_dphi.into();
        }
        res
    }
}
