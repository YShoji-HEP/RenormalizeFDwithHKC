use ndarray::{s, Array1};

#[cfg(test)]
mod tests {
    #[test]
    fn slices() {
        let a = ndarray::arr2(&[[0., 1., 2.], [1., 3., 5.], [3., 2., 7.]]);
        dbg!(&a);
        dbg!(&a.slice_axis(ndarray::Axis(1), ndarray::Slice::from(-2..-1)));
    }
}

pub trait Integrator {
    fn simpson(&self, dx: f64) -> f64;
}

impl Integrator for Array1<f64> {
    fn simpson(&self, dx: f64) -> f64 {
        let (y, rest) = if self.len() % 2 == 0 {
            (
                self.slice(s![..-1]),
                (self[self.len() - 1] + self[self.len() - 2]) / 2. * dx,
            )
        } else {
            (self.view(), 0.)
        };
        let sum_edge = y[0] + y[y.len() - 1];
        let sum_even: f64 = y.slice(s![2..-1;2]).sum();
        let sum_odd: f64 = y.slice(s![1..;2]).sum();
        dx / 3. * (sum_edge + 2. * sum_even + 4. * sum_odd) + rest
    }
}

pub mod stepper {
    use ndarray::{Array, Dimension};
    pub fn dp45<D: Dimension>(
        t: f64,
        f: &Array<f64, D>,
        dt: f64,
        dfdt: &dyn Fn(f64, &Array<f64, D>) -> Array<f64, D>,
    ) -> (Array<f64, D>, Array<f64, D>) {
        let d1 = dfdt(t, f) * dt;
        let t1 = t + 1. / 5. * dt;
        let f1 = 1. / 5. * &d1 + f;

        let d2 = dfdt(t1, &f1) * dt;
        let t2 = t + 3. / 10. * dt;
        let f2 = 3. / 40. * &d1 + 9. / 40. * &d2 + f;

        let d3 = dfdt(t2, &f2) * dt;
        let t3 = t + 4. / 5. * dt;
        let f3 = 44. / 45. * &d1 - 56. / 15. * &d2 + 32. / 9. * &d3 + f;

        let d4 = dfdt(t3, &f3) * dt;
        let t4 = t + 8. / 9. * dt;
        let f4 = 19372. / 6561. * &d1 - 25360. / 2187. * &d2 + 64448. / 6561. * &d3
            - 212. / 729. * &d4
            + f;

        let d5 = dfdt(t4, &f4) * dt;
        let t5 = t + dt;
        let f5 = 9017. / 3168. * &d1 - 355. / 33. * &d2 + 46732. / 5247. * &d3 + 49. / 176. * &d4
            - 5103. / 18656. * &d5
            + f;

        let d6 = dfdt(t5, &f5) * dt;
        let dy = 35. / 384. * &d1 + 500. / 1113. * &d3 + 125. / 192. * &d4 - 2187. / 6784. * &d5
            + 11. / 84. * &d6;
        let f6 = &dy + f;
        let t6 = t5;

        let d7 = dfdt(t6, &f6) * dt;
        let err = 71. / 57600. * d1 - 71. / 16695. * d3 + 71. / 1920. * d4 - 17253. / 339200. * d5
            + 22. / 525. * d6
            - 1. / 40. * d7;
        (dy, err)
    }
}
