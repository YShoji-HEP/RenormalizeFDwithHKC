use crate::bounce::Bounce;
use crate::potential::Potential;
use crate::tools::f128tools::*;
use crate::tools::stepper;
use ndarray::{arr1, Array1};
use ndarray_stats::QuantileExt;

const ABS_TOL: f128 = 1e-5;
const REL_TOL: f128 = 1e-5;

impl<T: Potential> Bounce<T> {
    pub fn ratio(&mut self, nu: f128, drho: f128) -> [f128; 3] {
        let mut rho = drho * 10.;
        let mut y = arr1(&[self.phi_0, 0., 1., 0., 1., 0., 0., 0., 0., 0.]);
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

            let ddphi = self.eom(rho_ode, phi, dphi);

            let ddpsi_nu = -(1. + 2. * nu) / rho_ode * dpsi_nu + self.v.second_deriv(phi) * psi_nu;
            let ddpsi0_nu =
                -(1. + 2. * nu) / rho_ode * dpsi0_nu + self.v.second_deriv_fv() * psi0_nu;
            let ddpsi1_nu = -(1. + 2. * nu) / rho_ode * dpsi1_nu
                + self.v.second_deriv_fv() * psi1_nu
                + (self.v.second_deriv(phi) - self.v.second_deriv_fv()) * psi0_nu;
            let ddpsi2_nu = -(1. + 2. * nu) / rho_ode * dpsi2_nu
                + self.v.second_deriv_fv() * psi2_nu
                + (self.v.second_deriv(phi) - self.v.second_deriv_fv()) * psi1_nu;
            arr1(&[
                dphi, ddphi, dpsi_nu, ddpsi_nu, dpsi0_nu, ddpsi0_nu, dpsi1_nu, ddpsi1_nu, dpsi2_nu,
                ddpsi2_nu,
            ])
        };
        let mut res_rho = vec![rho];
        let mut res_phi = vec![y[0]];
        let mut res_dphi = vec![y[1]];
        let mut res_psi_nu = vec![y[2]];
        let mut res_psi0_nu = vec![y[4]];
        let mut res_psi1_nu = vec![y[6]];
        let mut res_psi2_nu = vec![y[8]];
        let mut res_err = vec![0.];
        let mut dphi_max = -1.0_f128 / 0.0_f128;
        loop {
            let (dy, err) = stepper::dp45(rho, &y, drho, &dydrho);

            let err = *(err.map(|x| x.abs()) / (y.map(|x| x.abs()).mul(REL_TOL).add(ABS_TOL)))
                .max()
                .unwrap();
            y += &dy;
            rho += drho;

            let phi = y[0];
            let dphi = y[1];

            res_rho.push(rho);
            res_phi.push(y[0]);
            res_dphi.push(y[1]);

            res_psi_nu.push(y[2]);
            res_psi0_nu.push(y[4]);
            res_psi1_nu.push(y[6]);
            res_psi2_nu.push(y[8]);

            res_err.push(err);

            if dphi_max < dphi.abs() {
                dphi_max = dphi.abs();
            }
            if (phi - self.v.phi_fv()).abs() < (self.v.phi_top() - self.v.phi_fv()).abs() {
                if (phi - self.v.phi_fv()).abs()
                    < 0.0001 * (self.v.phi_top() - self.v.phi_fv()).abs()
                    && dphi.abs() < 0.0001 * dphi_max
                {
                    break;
                }
            }
        }
        self.rho = res_rho.into();
        self.phi = res_phi.into();
        self.phi_deriv = res_dphi.into();
        self.psi_nu = res_psi_nu.into();
        self.psi0_nu = res_psi0_nu.into();
        self.psi1_nu = res_psi1_nu.into();
        self.psi2_nu = res_psi2_nu.into();
        self.err = res_err.into();
        [y[2] / y[4], y[6] / y[4], y[8] / y[4]]
    }
}
