use crate::potential::*;
use crate::tools::*;
use ndarray::{arr1, Array1};

#[derive(Debug, PartialEq)]
pub enum ShootingResult {
    OverShoot,
    UnderShoot,
    Success,
}

pub struct Bounce<T: Potential> {
    pub v: T,
    pub dim: f128,
    pub phi_0: f128,
    pub rho: Array1<f128>,
    pub phi: Array1<f128>,
    pub phi_deriv: Array1<f128>,
    pub psi_nu: Array1<f128>,
    pub psi0_nu: Array1<f128>,
    pub psi1_nu: Array1<f128>,
    pub psi2_nu: Array1<f128>,
    pub err: Array1<f128>,
}

impl<T: Potential> Bounce<T> {
    pub fn new(v: T, dim: f128) -> Self {
        Self {
            v,
            dim,
            rho: Default::default(),
            phi_0: Default::default(),
            phi: Default::default(),
            phi_deriv: Default::default(),
            psi_nu: Default::default(),
            psi0_nu: Default::default(),
            psi1_nu: Default::default(),
            psi2_nu: Default::default(),
            err: Default::default(),
        }
    }
}

impl<T: Potential> Bounce<T> {
    pub fn find_profile(&mut self, drho: f128, max_iter: usize) {
        let mut phi_0_range = self.v.phi_top()..self.v.phi_tv();
        let mut i = 0;
        while i < max_iter {
            self.phi_0 = (phi_0_range.start + phi_0_range.end) / 2.;
            match self.shoot(drho, self.phi_0) {
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
    pub fn eom(&self, rho: f128, phi: f128, dphi: f128) -> f128 {
        self.v.first_deriv(phi) - (self.dim - 1.) / rho * dphi
    }
    pub fn shoot(&mut self, drho: f128, phi_ini: f128) -> ShootingResult {
        let mut rho = drho;
        let mut y = arr1(&[phi_ini, 0.]);
        let dydrho = |rho_ode: f128, fld: &Array1<f128>| {
            let phi = fld[0];
            let dphi = fld[1];
            let ddphi = self.eom(rho_ode, phi, dphi);
            arr1(&[dphi, ddphi])
        };
        let mut dphi_max = -1.0_f128 / 0.0_f128;
        let res = loop {
            let (dy, _) = stepper::dp45(rho, &y, drho, &dydrho);

            y += &dy;
            rho += drho;

            let phi = y[0];
            let dphi = y[1];

            if dphi_max < dphi.abs() {
                dphi_max = dphi.abs();
            }
            if (phi - self.v.phi_fv()).abs() < (self.v.phi_top() - self.v.phi_fv()).abs() {
                if (phi - self.v.phi_fv()).abs()
                    < 0.0001 * (self.v.phi_top() - self.v.phi_fv()).abs()
                    && dphi.abs() < 0.0001 * dphi_max
                {
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
