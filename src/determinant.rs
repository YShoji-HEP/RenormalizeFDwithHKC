use crate::bounce::Bounce;
use crate::potential::Potential;
use crate::tools::stepper;
use ndarray::{arr1, Array1};

// use crate::tools::f128tools::*;
// use ndarray_stats::QuantileExt;
// const ABS_TOL: f128 = 1e-5;
// const REL_TOL: f128 = 1e-5;

impl<T: Potential + Clone> Bounce<T> {
    pub fn ratio(&mut self, nu: f128, rho_ini: f128, step: f128) -> Array1<f128> {
        let mut rho = rho_ini;
        let phi_a = self.phi_0
            + self.v.first_deriv(self.phi_0) * rho_ini.powi(2) / 2. / self.dim
            + self.v.first_deriv(self.phi_0) * self.v.second_deriv(self.phi_0) * rho_ini.powi(4)
                / 8.
                / self.dim
                / (self.dim + 2.);
        let dphi_a = self.v.first_deriv(self.phi_0) * rho_ini / self.dim
            + self.v.first_deriv(self.phi_0) * self.v.second_deriv(self.phi_0) * rho_ini.powi(3)
                / 2.
                / self.dim
                / (self.dim + 2.);
        let mut y = arr1(&[phi_a, dphi_a, 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.]);
        let dydrho = |rho_ode: f128, fld: &Array1<f128>| {
            let phi = fld[0];
            let dphi = fld[1];
            let psi_nu = fld[2];
            let dpsi_nu = fld[3];
            let psi0_nu = fld[4];
            let dpsi0_nu = fld[5];
            let psi1_nu = fld[6];
            let dpsi1_nu = fld[7];
            let psi2_nu = fld[8];
            let dpsi2_nu = fld[9];
            let psi3_nu = fld[10];
            let dpsi3_nu = fld[11];

            let ddphi = self.v.first_deriv(phi) - (self.dim - 1.) / rho_ode * dphi;

            let ddpsi_nu = -(1. + 2. * nu) / rho_ode * dpsi_nu + self.v.second_deriv(phi) * psi_nu;
            let ddpsi0_nu =
                -(1. + 2. * nu) / rho_ode * dpsi0_nu + self.v.second_deriv_fv() * psi0_nu;
            let ddpsi1_nu = -(1. + 2. * nu) / rho_ode * dpsi1_nu
                + self.v.second_deriv_fv() * psi1_nu
                + (self.v.second_deriv(phi) - self.v.second_deriv_fv()) * psi0_nu;
            let ddpsi2_nu = -(1. + 2. * nu) / rho_ode * dpsi2_nu
                + self.v.second_deriv_fv() * psi2_nu
                + (self.v.second_deriv(phi) - self.v.second_deriv_fv()) * psi1_nu;
            let ddpsi3_nu = -(1. + 2. * nu) / rho_ode * dpsi3_nu
                + self.v.second_deriv_fv() * psi3_nu
                + (self.v.second_deriv(phi) - self.v.second_deriv_fv()) * psi2_nu;
            arr1(&[
                dphi, ddphi, dpsi_nu, ddpsi_nu, dpsi0_nu, ddpsi0_nu, dpsi1_nu, ddpsi1_nu, dpsi2_nu,
                ddpsi2_nu, dpsi3_nu, ddpsi3_nu,
            ])
        };

        loop {
            let drho = step / (1. / rho + (y[1] / self.phi_0).abs()).sqrt();
            let (dy, _err) = stepper::dp45(rho, &y, drho, &dydrho);
            y += &dy;
            rho += drho;

            if rho > self.rho_max {
                break;
            }

            // let err = *(err.map(|x| x.abs()) / (y.map(|x| x.abs()).mul(REL_TOL).add(ABS_TOL)))
            //     .max()
            //     .unwrap();
        }
        arr1(&[y[2] / y[4], y[6] / y[4], y[8] / y[4], y[10] / y[4]])
    }
}
