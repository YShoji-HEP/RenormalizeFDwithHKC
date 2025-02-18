use crate::bounce::Bounce;
use crate::potential::Potential;
use crate::tools::f128tools::*;
use crate::tools::stepper;
use ndarray::{arr1, Array1};
use ndarray_stats::QuantileExt;

const ABS_TOL: f128 = 1e-5;
const REL_TOL: f128 = 1e-5;

impl<T: Potential> Bounce<T> {
    pub fn hk(
        &mut self,
        nu: f128,
        drho: f128,
        i_nu: &dyn Fn(f128, f128) -> [f128; 5],
        debug: bool,
    ) -> [f128; 5] {
        let mut rho = drho * 10.;
        let mut y = arr1(&[self.phi_0, 0., 0., 0., 0., 0., 0.]);
        let dydrho = |rho_ode: f128, fld: &Array1<f128>| {
            let phi = fld[0];
            let dphi = fld[1];
            let int_nu = i_nu(nu, rho_ode);

            let ddphi = self.v.first_deriv(phi) - (self.dim - 1.) / rho_ode * dphi;
            let d3phi = self.v.second_deriv(phi) * dphi - (self.dim - 1.) / rho_ode * ddphi
                + (self.dim - 1.) / rho_ode.powi(2) * dphi;
            let d4phi = self.v.third_deriv(phi) * dphi.powi(2) + self.v.second_deriv(phi) * ddphi
                - (self.dim - 1.) / rho_ode * d3phi
                + 2. * (self.dim - 1.) / rho_ode.powi(2) * ddphi
                - 2. * (self.dim - 1.) / rho_ode.powi(3) * dphi;
            let d5phi = self.v.forth_deriv(phi) * dphi.powi(3)
                + 3. * self.v.third_deriv(phi) * ddphi * dphi
                + self.v.second_deriv(phi) * d3phi
                - (self.dim - 1.) / rho_ode * d4phi
                + 3. * (self.dim - 1.) / rho_ode.powi(2) * d3phi
                - 6. * (self.dim - 1.) / rho_ode.powi(3) * ddphi
                + 6. * (self.dim - 1.) / rho_ode.powi(4) * dphi;

            let m2_hat = self.v.second_deriv_fv();
            let m2 = self.v.second_deriv(phi);
            let dm2 = self.v.third_deriv(phi) * dphi;
            let ddm2 = self.v.third_deriv(phi) * ddphi + self.v.forth_deriv(phi) * dphi.powi(2);
            let d3m2 =
                self.v.third_deriv(phi) * d3phi + 3. * self.v.forth_deriv(phi) * ddphi * dphi;
            let d4m2 = self.v.third_deriv(phi) * d4phi
                + 4. * self.v.forth_deriv(phi) * d3phi * dphi
                + 3. * self.v.forth_deriv(phi) * ddphi.powi(2);
            let d5m2 = self.v.third_deriv(phi) * d5phi
                + 5. * self.v.forth_deriv(phi) * d4phi * dphi
                + 4. * self.v.forth_deriv(phi) * d3phi * ddphi
                + 6. * self.v.forth_deriv(phi) * ddphi * d3phi;

            let dhkc1 = (m2 - m2_hat) * int_nu[0];
            let dhkc2 = -1. / 2. * (m2.powi(2) - m2_hat.powi(2)) * int_nu[1];
            let dhkc3 = 1. / 6. * (m2.powi(3) + 1. / 2. * dm2.powi(2) - m2_hat.powi(3)) * int_nu[2];
            let dhkc4 = 0. * int_nu[3];
            let dhkc5 = 0. * d5m2 * int_nu[4];

            arr1(&[dphi, ddphi, dhkc1, dhkc2, dhkc3, dhkc4, dhkc5])
        };
        let mut res_rho = vec![rho];
        let mut res_phi = vec![y[0]];
        let mut res_dphi = vec![y[1]];
        let mut res_dhkc1 = vec![y[2]];
        let mut res_dhkc2 = vec![y[3]];
        let mut res_dhkc3 = vec![y[4]];
        let mut res_dhkc4 = vec![y[5]];
        let mut res_dhkc5 = vec![y[6]];
        let mut res_err = vec![0.];

        loop {
            let (dy, err) = stepper::dp45(rho, &y, drho, &dydrho);
            y += &dy;
            rho += drho;

            if rho > self.rho_max {
                break;
            }

            if debug {
                let err = *(err.map(|x| x.abs()) / (y.map(|x| x.abs()).mul(REL_TOL).add(ABS_TOL)))
                    .max()
                    .unwrap();
                res_rho.push(rho);
                res_phi.push(y[0]);
                res_dphi.push(y[1]);

                res_dhkc1.push(y[2]);
                res_dhkc2.push(y[3]);
                res_dhkc3.push(y[4]);
                res_dhkc4.push(y[5]);
                res_dhkc5.push(y[6]);

                res_err.push(err);
            }
        }
        if debug {
            dbgbb::dbgbb!(res_dhkc1
                .iter()
                .map(|&x| x as f64)
                .collect::<Vec<_>>()
                .rename("hke1"));
            dbgbb::dbgbb!(res_dhkc2
                .iter()
                .map(|&x| x as f64)
                .collect::<Vec<_>>()
                .rename("hke2"));
            dbgbb::dbgbb!(res_dhkc3
                .iter()
                .map(|&x| x as f64)
                .collect::<Vec<_>>()
                .rename("hke3"));
        }
        [y[2], y[3], y[4], y[5], y[6]]
    }
}
