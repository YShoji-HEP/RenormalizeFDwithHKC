use crate::bounce::Bounce;
use crate::potential::Potential;
use crate::tools::stepper;
use ndarray::{arr1, Array1};

// use ndarray_stats::QuantileExt;
// use crate::tools::f128tools::*;
// const ABS_TOL: f128 = 1e-5;
// const REL_TOL: f128 = 1e-5;

impl<T: Potential + Clone> Bounce<T> {
    pub fn hk(
        &mut self,
        nu: f128,
        i_nu: &dyn Fn(f128, f128) -> [f128; 5],
        z: f128,
        rho_ini: f128,
        step: f128,
    ) -> Array1<f128> {
        let mut rho = rho_ini;
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

            let m2_hat = self.v.second_deriv_fv() - z;
            let m2 = self.v.second_deriv(phi) - z;
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
            let dhkc3 = 1. / 6. * (m2.powi(3) - m2_hat.powi(3) + 1. / 2. * dm2.powi(2)) * int_nu[2];
            let dhkc4 = -1. / 24.
                * (m2.powi(4) - m2_hat.powi(4)
                    + 2. * m2 * dm2.powi(2)
                    + 4. / 15. * (ddm2.powi(2) + (self.dim - 1.) * dm2.powi(2) / rho_ode.powi(2))
                    + 1. / 15.
                        * dm2
                        * (d3m2 + (self.dim - 1.) / rho_ode * ddm2
                            - (self.dim - 1.) / rho_ode.powi(2) * dm2))
                * int_nu[3];
            let dhkc5 = 1. / 120.
                * (m2.powi(5) - m2_hat.powi(5)
                    + 5. * m2.powi(2) * dm2.powi(2)
                    + 13. / 6. * ddm2 * dm2.powi(2)
                    + 5. / 4.
                        * m2
                        * (ddm2.powi(2) + (self.dim - 1.) / rho_ode.powi(2) * dm2.powi(2))
                    + 1. / 8. * dm2.powi(2) * (ddm2 + (self.dim - 1.) / rho_ode * dm2)
                    + 1. / 4.
                        * m2
                        * dm2
                        * (d3m2 + (self.dim - 1.) / rho_ode * ddm2
                            - (self.dim - 1.) / rho_ode.powi(2) * dm2)
                    + 1. / 7.
                        * (d3m2.powi(2)
                            + 3. * (self.dim - 1.) / rho_ode.powi(4)
                                * (rho_ode * ddm2 - dm2).powi(2))
                    + 5. / 56.
                        * (d4m2 * ddm2
                            + (self.dim - 1.) / rho_ode * d3m2 * ddm2
                            + (self.dim - 1.) / rho_ode.powi(2) * dm2 * d3m2
                            - 2. * (self.dim - 1.) / rho_ode.powi(2) * ddm2.powi(2)
                            + (self.dim.powi(2) - 1.) / rho_ode.powi(3) * ddm2 * dm2
                            - (self.dim - 1.).powi(2) / rho_ode.powi(4) * dm2.powi(2))
                    + 1. / 112.
                        * (d3m2 + (self.dim - 1.) / rho_ode * ddm2
                            - (self.dim - 1.) / rho_ode.powi(2) * dm2)
                            .powi(2)
                    + 1. / 112.
                        * (d5m2 * dm2
                            + 2. * (self.dim - 1.) / rho_ode * dm2 * d4m2
                            + (self.dim - 1.) * (self.dim - 5.) / rho_ode.powi(2) * dm2 * d3m2
                            - 3. * (self.dim - 1.) * (self.dim - 3.) / rho_ode.powi(3)
                                * dm2
                                * ddm2
                            + 3. * (self.dim - 1.) * (self.dim - 3.) / rho_ode.powi(4)
                                * dm2.powi(2))
                    + z / 12.
                        * ((ddm2.powi(2) + (self.dim - 1.) * dm2.powi(2) / rho_ode.powi(2))
                            + dm2
                                * (d3m2 + (self.dim - 1.) / rho_ode * ddm2
                                    - (self.dim - 1.) / rho_ode.powi(2) * dm2)))
                * int_nu[4];

            arr1(&[dphi, ddphi, dhkc1, dhkc2, dhkc3, dhkc4, dhkc5])
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
        arr1(&[y[2], y[3], y[4], y[5], y[6]])
    }
}
