
#[cfg(test)]
mod tests {
    #[test]
    fn slices() {
        let a = ndarray::arr2(&[[0., 1., 2.], [1., 3., 5.], [3., 2., 7.]]);
        dbg!(&a);
        dbg!(&a.slice_axis(ndarray::Axis(1), ndarray::Slice::from(-2..-1)));
    }
}

// pub trait Integrator {
//     fn simpson(&self, dx: f128) -> f128;
// }

// impl Integrator for Array1<f128> {
//     fn simpson(&self, dx: f128) -> f128 {
//         let (y, rest) = if self.len() % 2 == 0 {
//             (
//                 self.slice(s![..-1]),
//                 (self[self.len() - 1] + self[self.len() - 2]) / 2. * dx,
//             )
//         } else {
//             (self.view(), 0.)
//         };
//         let sum_edge = y[0] + y[y.len() - 1];
//         let sum_even: f128 = y.slice(s![2..-1;2]).sum();
//         let sum_odd: f128 = y.slice(s![1..;2]).sum();
//         dx / 3. * (sum_edge + 2. * sum_even + 4. * sum_odd) + rest
//     }
// }

pub mod f128tools {
    use ndarray::{Array, Dimension};
    pub trait Mul128 {
        fn mul(&self, z: f128) -> Self;
    }
    impl<D: Dimension> Mul128 for Array<f128, D> {
        fn mul(&self, z: f128) -> Self {
            self.map(|x| x * z)
        }
    }
    pub trait Add128 {
        fn add(&self, z: f128) -> Self;
    }
    impl<D: Dimension> Add128 for Array<f128, D> {
        fn add(&self, z: f128) -> Self {
            self.map(|x| x + z)
        }
    }
}

pub mod stepper {
    use ndarray::{Array, Dimension};
    use super::f128tools::*;
    pub fn dp45<D: Dimension>(
        t: f128,
        f: &Array<f128, D>,
        dt: f128,
        dfdt: &dyn Fn(f128, &Array<f128, D>) -> Array<f128, D>,
    ) -> (Array<f128, D>, Array<f128, D>) {
        let d1 = dfdt(t, f).mul(dt);
        let t1 = t + 1. / 5. * dt;
        let f1 = d1.mul(1. / 5.) + f;

        let d2 = dfdt(t1, &f1).mul(dt);
        let t2 = t + 3. / 10. * dt;
        let f2 = d1.mul(3. / 40.) + d2.mul(9. / 40.) + f;

        let d3 = dfdt(t2, &f2).mul(dt);
        let t3 = t + 4. / 5. * dt;
        let f3 = d1.mul(44. / 45.) - d2.mul(56. / 15.) + d3.mul(32. / 9.) + f;

        let d4 = dfdt(t3, &f3).mul(dt);
        let t4 = t + 8. / 9. * dt;
        let f4 = d1.mul(19372. / 6561.) - d2.mul(25360. / 2187.) + d3.mul(64448. / 6561.)
            - d4.mul(212. / 729.)
            + f;

        let d5 = dfdt(t4, &f4).mul(dt);
        let t5 = t + dt;
        let f5 = d1.mul(9017. / 3168.) - d2.mul(355. / 33.)
            + d3.mul(46732. / 5247.)
            + d4.mul(49. / 176.)
            - d5.mul(5103. / 18656.)
            + f;

        let d6 = dfdt(t5, &f5).mul(dt);
        let dy = d1.mul(35. / 384.) + d3.mul(500. / 1113.) + d4.mul(125. / 192.)
            - d5.mul(2187. / 6784.)
            + d6.mul(11. / 84.);
        let f6 = &dy + f;
        let t6 = t5;

        let d7 = dfdt(t6, &f6).mul(dt);
        let err = d1.mul(71. / 57600.) - d3.mul(71. / 16695.) + d4.mul(71. / 1920.)
            - d5.mul(17253. / 339200.)
            + d6.mul(22. / 525.)
            - d7.mul(1. / 40.);
        (dy, err)
    }
}
