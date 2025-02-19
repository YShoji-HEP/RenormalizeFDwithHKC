use crate::potential::*;
use crate::tools::*;
use ndarray::{arr1, Array1};

#[derive(Debug, PartialEq)]
pub enum ShootingResult {
    OverShoot,
    UnderShoot,
    Success,
}

#[derive(Clone)]
pub struct Bounce<T: Potential + Clone> {
    pub v: T,
    pub dim: f128,
    pub phi_0: f128,
    pub rho_max: f128,
}

impl<T: Potential + Clone> Bounce<T> {
    pub fn new(v: T, dim: f128) -> Self {
        Self {
            v,
            dim,
            phi_0: Default::default(),
            rho_max: 0.,
        }
    }
}

impl<T: Potential + Clone> Bounce<T> {
    pub fn find_profile(&mut self, tol: f128, max_iter: usize, rho_ini: f128, step: f128) {
        let mut phi_0_range = self.v.phi_top()..self.v.phi_tv();
        let mut i = 0;
        while i < max_iter {
            self.phi_0 = (phi_0_range.start + phi_0_range.end) / 2.;
            dbg!(i, self.phi_0 as f64);
            match self.shoot(self.phi_0, tol, rho_ini, step) {
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
            }
        }
    }
    pub fn shoot(&mut self, phi_ini: f128, tol: f128, rho_ini: f128, step: f128) -> ShootingResult {
        let mut rho = rho_ini;
        let mut y = arr1(&[phi_ini, 0.]);
        let dydrho = |rho_ode: f128, fld: &Array1<f128>| {
            let phi = fld[0];
            let dphi = fld[1];
            let ddphi = self.v.first_deriv(phi) - (self.dim - 1.) / rho_ode * dphi;
            arr1(&[dphi, ddphi])
        };
        let mut dphi_max = -1.0_f128 / 0.0_f128;
        let res = loop {
            let drho = step / (1. / rho + (y[1] / self.phi_0).abs()).sqrt();
            let (dy, _) = stepper::dp45(rho, &y, drho, &dydrho);

            y += &dy;
            rho += drho;

            let phi = y[0];
            let dphi = y[1];

            if dphi_max < dphi.abs() {
                dphi_max = dphi.abs();
            }
            if (phi - self.v.phi_fv()).abs() < (self.v.phi_top() - self.v.phi_fv()).abs() {
                if (phi - self.v.phi_fv()).abs() < tol * (self.v.phi_top() - self.v.phi_fv()).abs()
                    && dphi.abs() < tol * dphi_max
                {
                    self.rho_max = rho;
                    break ShootingResult::Success;
                }
                if dphi > 0. {
                    break ShootingResult::UnderShoot;
                }
                if phi < self.v.phi_fv() {
                    break ShootingResult::OverShoot;
                }
            }
        };
        res
    }
}
