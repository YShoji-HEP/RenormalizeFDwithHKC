use crate::bounce::Bounce;
use crate::potential::Potential;
use crate::tools::stepper;
use ndarray::{Array1, arr1};

impl<T: Potential + Clone> Bounce<T> {
    pub fn rwkb(&mut self, nu: usize, rho_ini: f128, step: f128) -> Array1<f128> {
        let nu: i64 = nu.try_into().unwrap();
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
        let mut y = arr1(&[phi_a, dphi_a, 0., 0., 0., 0., 0., 0., 0., 0.]);
        let dydrho = |rho_ode: f128, fld: &Array1<f128>| {
            let phi = fld[0];
            let dphi = fld[1];

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
            let d6phi = 6. * self.v.forth_deriv(phi) * dphi.powi(2) * ddphi
                + 4. * self.v.third_deriv(phi) * d3phi * dphi
                + 3. * self.v.third_deriv(phi) * ddphi.powi(2)
                + self.v.second_deriv(phi) * d4phi
                - (self.dim - 1.) / rho_ode * d5phi
                + 4. * (self.dim - 1.) / rho_ode.powi(2) * d4phi
                - 12. * (self.dim - 1.) / rho_ode.powi(3) * d3phi
                + 24. * (self.dim - 1.) / rho_ode.powi(4) * ddphi
                - 24. * (self.dim - 1.) / rho_ode.powi(5) * dphi;

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
                + 10. * self.v.forth_deriv(phi) * d3phi * ddphi;
            let d6m2 = self.v.third_deriv(phi) * d6phi
                + 6. * self.v.forth_deriv(phi) * d5phi * dphi
                + 15. * self.v.forth_deriv(phi) * d4phi * ddphi
                + 10. * self.v.forth_deriv(phi) * d3phi.powi(2);

            let u = |lmax: f128| (1.0 + m2_hat * rho_ode.powi(2) / lmax.powi(2)).sqrt();

            let mu = 1.;

            let q_log = |lmax: f128| {
                let lmax = lmax.max(1.0);
                -(1.0 / 24.0)
                    * (mu * rho_ode / ((u(lmax) + 1.0) * lmax)).ln()
                    * (6.0 * m2_hat * rho_ode.powi(3) * (m2 - m2_hat)
                        + 3.0 * rho_ode.powi(3) * (m2 - m2_hat).powi(2)
                        - 3.0 * rho_ode.powi(2) * dm2
                        - rho_ode.powi(3) * ddm2)
            };

            let q_2 = |lmax: f128| {
                -(1.0 / 8.0) * (u(lmax).powi(2) + 2.0 * u(lmax) - 1.0) * rho_ode * (m2 - m2_hat)
            };

            let q_1 = |lmax: f128| -(3.0 / (4.0 * u(lmax))) * rho_ode * (m2 - m2_hat);

            let q_0 = |lmax: f128| {
                let term = -6.0 * rho_ode.powi(3) * u(lmax).powi(4) * (m2 - m2_hat).powi(2)
                    + 2.0 * rho_ode.powi(3) * u(lmax).powi(4) * ddm2
                    + 2.0 * rho_ode.powi(2) * (3.0 * u(lmax).powi(2) + 1.0) * u(lmax).powi(2) * dm2
                    + rho_ode
                        * (-52.0 * u(lmax).powi(4) + 25.0 * u(lmax).powi(2) + 3.0)
                        * (m2 - m2_hat);
                term / (48.0 * u(lmax).powi(5))
            };

            let q_m1 = |lmax: f128| {
                let term = 6.0 * rho_ode.powi(3) * u(lmax).powi(4) * (m2 - m2_hat).powi(2)
                    - 2.0 * rho_ode.powi(3) * u(lmax).powi(4) * ddm2
                    - 6.0 * rho_ode.powi(2) * u(lmax).powi(2) * dm2
                    - rho_ode
                        * (16.0 * u(lmax).powi(6) - 37.0 * u(lmax).powi(4)
                            + 6.0 * u(lmax).powi(2)
                            + 15.0)
                        * (m2 - m2_hat);
                term / (32.0 * u(lmax).powi(7))
            };

            let q_m2 = |lmax: f128| {
                let term = 240.0
                    * rho_ode.powi(5)
                    * (u(lmax).powi(3) - 1.0)
                    * u(lmax).powi(8)
                    * (m2 - m2_hat).powi(3)
                    - 120.0
                        * rho_ode.powi(5)
                        * (u(lmax).powi(3) - 1.0)
                        * u(lmax).powi(8)
                        * dm2.powi(2)
                    + 24.0 * rho_ode.powi(5) * (u(lmax).powi(3) - 1.0) * u(lmax).powi(8) * d4m2
                    + 144.0 * rho_ode.powi(4) * (u(lmax).powi(5) - 1.0) * u(lmax).powi(6) * d3m2
                    + 60.0
                        * rho_ode.powi(3)
                        * (52.0 * u(lmax).powi(6) - 127.0 * u(lmax).powi(4)
                            + 60.0 * u(lmax).powi(2)
                            + 15.0)
                        * u(lmax).powi(4)
                        * (m2 - m2_hat).powi(2)
                    + 4.0
                        * rho_ode.powi(3)
                        * (18.0 * u(lmax).powi(7) - 260.0 * u(lmax).powi(6)
                            + 635.0 * u(lmax).powi(4)
                            - 228.0 * u(lmax).powi(2)
                            - 165.0)
                        * u(lmax).powi(4)
                        * ddm2
                    - 720.0
                        * rho_ode.powi(4)
                        * (u(lmax).powi(5) - 1.0)
                        * u(lmax).powi(6)
                        * (m2 - m2_hat)
                        * dm2
                    - 12.0
                        * rho_ode.powi(2)
                        * (6.0 * u(lmax).powi(9) + 510.0 * u(lmax).powi(6)
                            - 1011.0 * u(lmax).powi(4)
                            + 320.0 * u(lmax).powi(2)
                            + 175.0)
                        * u(lmax).powi(2)
                        * dm2
                    - 240.0
                        * rho_ode.powi(5)
                        * (u(lmax).powi(3) - 1.0)
                        * u(lmax).powi(8)
                        * (m2 - m2_hat)
                        * ddm2
                    + 3.0
                        * rho_ode
                        * (u(lmax).powi(2) - 1.0).powi(2)
                        * (3288.0 * u(lmax).powi(6) + 1605.0 * u(lmax).powi(4)
                            - 7630.0 * u(lmax).powi(2)
                            - 1575.0)
                        * (m2 - m2_hat);
                term / (11520.0 * u(lmax).powi(11) * (u(lmax).powi(2) - 1.0))
            };

            let q_m3 = |lmax: f128| {
                let term = -720.0 * rho_ode.powi(5) * u(lmax).powi(8) * (m2 - m2_hat).powi(3)
                    + 360.0 * rho_ode.powi(5) * u(lmax).powi(8) * dm2.powi(2)
                    - 72.0 * rho_ode.powi(5) * u(lmax).powi(8) * d4m2
                    - 720.0 * rho_ode.powi(4) * u(lmax).powi(6) * d3m2
                    + 60.0
                        * rho_ode.powi(3)
                        * (16.0 * u(lmax).powi(6) - 111.0 * u(lmax).powi(4)
                            + 30.0 * u(lmax).powi(2)
                            + 105.0)
                        * u(lmax).powi(4)
                        * (m2 - m2_hat).powi(2)
                    - 20.0
                        * rho_ode.powi(3)
                        * (16.0 * u(lmax).powi(6)
                            - 111.0 * u(lmax).powi(4)
                            - 42.0 * u(lmax).powi(2)
                            + 231.0)
                        * u(lmax).powi(4)
                        * ddm2
                    - 60.0
                        * rho_ode.powi(2)
                        * (90.0 * u(lmax).powi(6)
                            - 201.0 * u(lmax).powi(4)
                            - 182.0 * u(lmax).powi(2)
                            + 315.0)
                        * u(lmax).powi(2)
                        * dm2
                    + 720.0 * rho_ode.powi(5) * u(lmax).powi(8) * (m2 - m2_hat) * ddm2
                    + 3600.0 * rho_ode.powi(4) * u(lmax).powi(6) * (m2 - m2_hat) * dm2
                    + 15.0
                        * rho_ode
                        * (96.0 * u(lmax).powi(10) + 63.0 * u(lmax).powi(8)
                            - 2980.0 * u(lmax).powi(6)
                            + 3010.0 * u(lmax).powi(4)
                            + 3276.0 * u(lmax).powi(2)
                            - 3465.0)
                        * (m2 - m2_hat);

                term / (7680.0 * u(lmax).powi(13))
            };

            let q_m4 = |lmax: f128| {
                let term1 = rho_ode.powi(7)
                    * (2.0 * u(lmax).powi(5) - 5.0 * u(lmax).powi(2) + 3.0)
                    * u(lmax).powi(12)
                    * (-1680.0 * (m2 - m2_hat).powi(4) + 3360.0 * (m2 - m2_hat) * dm2.powi(2)
                        - 1008.0 * ddm2.powi(2)
                        + 3360.0 * (m2 - m2_hat).powi(2) * ddm2
                        - 1344.0 * dm2 * d3m2
                        - 672.0 * (m2 - m2_hat) * d4m2
                        + 48.0 * d6m2);

                let term2 = rho_ode.powi(6)
                    * (2.0 * u(lmax).powi(7) - 7.0 * u(lmax).powi(2) + 5.0)
                    * u(lmax).powi(10)
                    * (432.0 * d5m2 - 7392.0 * dm2 * ddm2 - 4032.0 * (m2 - m2_hat) * d3m2
                        + 10080.0 * (m2 - m2_hat).powi(2) * dm2)
                    - 1680.0
                        * rho_ode.powi(5)
                        * (u(lmax).powi(2) - 1.0).powi(2)
                        * (52.0 * u(lmax).powi(4) - 125.0 * u(lmax).powi(2) - 35.0)
                        * u(lmax).powi(8)
                        * (m2 - m2_hat).powi(3);

                let term3 = 5.0
                    * rho_ode
                    * (u(lmax).powi(2) - 1.0).powi(3)
                    * (64692.0 * u(lmax).powi(10)
                        - 748223.0 * u(lmax).powi(8)
                        - 1201788.0 * u(lmax).powi(6)
                        + 7638246.0 * u(lmax).powi(4)
                        - 4708704.0 * u(lmax).powi(2)
                        - 1576575.0)
                    * (m2 - m2_hat);

                let term4 = -168.0
                    * rho_ode.powi(5)
                    * (28.0 * u(lmax).powi(9) - 260.0 * u(lmax).powi(8) + 1145.0 * u(lmax).powi(6)
                        - 1083.0 * u(lmax).powi(4)
                        - 355.0 * u(lmax).powi(2)
                        + 525.0)
                    * u(lmax).powi(8)
                    * dm2.powi(2);

                let term5 = -336.0
                    * rho_ode.powi(5)
                    * (12.0 * u(lmax).powi(9) - 260.0 * u(lmax).powi(8) + 1145.0 * u(lmax).powi(6)
                        - 1167.0 * u(lmax).powi(4)
                        - 115.0 * u(lmax).powi(2)
                        + 385.0)
                    * u(lmax).powi(8)
                    * (m2 - m2_hat)
                    * ddm2;

                let term6 = 24.0
                    * rho_ode.powi(5)
                    * (36.0 * u(lmax).powi(9) - 364.0 * u(lmax).powi(8) + 1603.0 * u(lmax).powi(6)
                        - 1365.0 * u(lmax).powi(4)
                        - 785.0 * u(lmax).powi(2)
                        + 875.0)
                    * u(lmax).powi(8)
                    * d4m2;

                let term7 = 1008.0
                    * rho_ode.powi(4)
                    * (4.0 * u(lmax).powi(11) + 850.0 * u(lmax).powi(8) - 2869.0 * u(lmax).powi(6)
                        + 2645.0 * u(lmax).powi(4)
                        - 105.0 * u(lmax).powi(2)
                        - 525.0)
                    * u(lmax).powi(6)
                    * (m2 - m2_hat)
                    * dm2;

                let term8 = -144.0
                    * rho_ode.powi(4)
                    * (8.0 * u(lmax).powi(11) + 1190.0 * u(lmax).powi(8)
                        - 3843.0 * u(lmax).powi(6)
                        + 3030.0 * u(lmax).powi(4)
                        + 665.0 * u(lmax).powi(2)
                        - 1050.0)
                    * u(lmax).powi(6)
                    * d3m2;

                let term9 = -42.0
                    * rho_ode.powi(3)
                    * (u(lmax).powi(2) - 1.0).powi(2)
                    * (9864.0 * u(lmax).powi(8)
                        - 8415.0 * u(lmax).powi(6)
                        - 64645.0 * u(lmax).powi(4)
                        + 54495.0 * u(lmax).powi(2)
                        + 17325.0)
                    * u(lmax).powi(4)
                    * (m2 - m2_hat).powi(2);

                let term10 = 2.0
                    * rho_ode.powi(3)
                    * (432.0 * u(lmax).powi(13) + 69048.0 * u(lmax).powi(12)
                        - 25641.0 * u(lmax).powi(10)
                        - 1242409.0 * u(lmax).powi(8)
                        + 2941010.0 * u(lmax).powi(6)
                        - 1987510.0 * u(lmax).powi(4)
                        - 167265.0 * u(lmax).powi(2)
                        + 412335.0)
                    * u(lmax).powi(4)
                    * ddm2;

                let term11 = -6.0
                    * rho_ode.powi(2)
                    * (144.0 * u(lmax).powi(15) + 46032.0 * u(lmax).powi(14)
                        - 228564.0 * u(lmax).powi(12)
                        - 659897.0 * u(lmax).powi(10)
                        + 4287475.0 * u(lmax).powi(8)
                        - 6809390.0 * u(lmax).powi(6)
                        + 3736110.0 * u(lmax).powi(4)
                        + 153615.0 * u(lmax).powi(2)
                        - 525525.0)
                    * u(lmax).powi(2)
                    * dm2;

                let term = term1
                    + term2
                    + term3
                    + term4
                    + term5
                    + term6
                    + term7
                    + term8
                    + term9
                    + term10
                    + term11;

                term / (645120.0 * u(lmax).powi(17) * (u(lmax).powi(2) - 1.0).powi(2))
            };

            let nu = nu as f128;
            arr1(&[
                dphi,
                ddphi,
                q_2(nu - 1.) * (nu - 1.).max(1.).powi(2) - q_2(nu - 2.) * (nu - 2.).max(1.).powi(2),
                q_1(nu - 1.) * (nu - 1.).max(1.) - q_1(nu - 2.) * (nu - 2.).max(1.),
                q_log(nu - 1.) - q_log(nu - 2.),
                q_0(nu - 1.) - q_0(nu - 2.),
                q_m1(nu - 1.) * (nu - 1.).max(1.).powi(-1)
                    - q_m1(nu - 2.) * (nu - 2.).max(1.).powi(-1),
                q_m2(nu - 1.) * (nu - 1.).max(1.).powi(-2)
                    - q_m2(nu - 2.) * (nu - 2.).max(1.).powi(-2),
                q_m3(nu - 1.) * (nu - 1.).max(1.).powi(-3)
                    - q_m3(nu - 2.) * (nu - 2.).max(1.).powi(-3),
                q_m4(nu - 1.) * (nu - 1.).max(1.).powi(-4)
                    - q_m4(nu - 2.) * (nu - 2.).max(1.).powi(-4),
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
        }
        arr1(&[y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]])
    }
}
