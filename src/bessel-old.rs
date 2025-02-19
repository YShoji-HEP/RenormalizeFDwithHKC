use std::collections::HashMap;
#[cfg(test)]
mod tests {
    use super::*;
    use dbgbb::dbgbb;
    use ndarray::Array1;
    #[test]
    fn compare() {
        let ik = BesselIK::new(20);
        let z: Array1<f128> = (0..100).map(|i| i as f128 / 100. * 50.).collect();
        // let lz = z.map(|&x| ik.low_z(x, 100) as f64);
        // let hz = z.map(|&x| ik.high_z(x, 30) as f64);
        let res = z.map(|&x| ik.deriv(x,2) as f64);
        let z = z.map(|&x| x as f64);
        dbgbb!(z, res);
    }
}

pub struct BesselIK {
    nu: usize,
    sum_harmonic_init: f128,
}

const EULER_GAMMA: f128 = 0.57721566490153286060651209008240243104215933593992;

impl BesselIK {
    pub fn new(nu: usize) -> Self {
        let sum_harmonic_init: f128 = (1..nu + 1)
            .map(|i| 1. / i as f128)
            .fold(0., |acc, x| acc + x);
        Self {
            nu,
            sum_harmonic_init: 0.,
        }
    }
    pub fn val(&self, z: f128) -> f128 {
        if z < 20. {
            self.low_z(z, 110)
        } else {
            self.high_z(z, 35)
        }
    }
    pub fn first_deriv(&self, z: f128) -> f128 {
        if z < 20. {
            self.low_z_first_deriv(z, 110)
        } else {
            self.high_z_first_deriv(z, 35)
        }
    }
    fn low_z(&self, z: f128, n: usize) -> f128 {
        let fact = |i: usize| ((i + 1) as f128).gamma();
        let mut res = 0.;
        for k in 0..n + 1 {
            let mut temp = 0.;
            for l in 0..(k + 1).min(self.nu) {
                temp += (-1f128).powi(l as i32) * fact(self.nu - l - 1)
                    / fact(l)
                    / fact(k - l)
                    / fact(k - l + self.nu);
            }
            res += temp * (z / 2.).powi(2 * k as i32) / 2.;
        }
        for k in 0..(n + 1).max(self.nu) - self.nu {
            let mut temp = 0.;
            let mut sum_harmonic = self.sum_harmonic_init;
            for l in 0..k + 1 {
                temp += (sum_harmonic / 2. - EULER_GAMMA - (z / 2.).ln())
                    / fact(l)
                    / fact(l + self.nu)
                    / fact(k - l)
                    / fact(k - l + self.nu);
                sum_harmonic += 1. / (self.nu + l + 1) as f128 + 1. / (l + 1) as f128;
            }
            res += temp * (z / 2.).powi(2 * (k + self.nu) as i32) * (-1f128).powi(self.nu as i32);
        }
        res
    }
    fn high_z(&self, z: f128, n: usize) -> f128 {
        let p = (1. + (z / self.nu as f128).powi(2)).powf(-0.5);
        let mut init = HashMap::new();
        init.insert(0, 1f128);
        let mut coefs = vec![init];

        let mut res = 0.;
        for k in 1..2 * n + 1 {
            let mut new_coef = HashMap::new();
            for (&i, c) in &coefs[k - 1] {
                let entry = new_coef.entry(i + 1).or_insert(0.);
                *entry += c * (2. * i as f128 + 1.).powi(2) / 8. / (i as f128 + 1.);
                let entry = new_coef.entry(i + 3).or_insert(0.);
                *entry -= c * (i as f128 / 2. + 5. / 8. / (3. + i as f128));
            }
            coefs.push(new_coef);
        }
        for k in 0..n + 1 {
            let mut temp = 0.;
            for m in 0..2 * k + 1 {
                let u_1 = coefs[m]
                    .iter()
                    .map(|(&i, c)| c * p.powi(i))
                    .fold(0., |acc, x| acc + x);
                let u_2 = coefs[2 * k - m]
                    .iter()
                    .map(|(&i, c)| c * p.powi(i))
                    .fold(0., |acc, x| acc + x);
                temp += (-1f128).powi(m as i32) * u_1 * u_2;
            }
            res += temp / (self.nu as f128).powi(2 * k as i32);
        }
        res * p / 2. / self.nu as f128
    }
    fn low_z_first_deriv(&self, z: f128, n: usize) -> f128 {
        let fact = |i: usize| ((i + 1) as f128).gamma();
        let mut res = 0.;
        for k in 1..n + 1 {
            let mut temp = 0.;
            for l in 0..(k + 1).min(self.nu) {
                temp += (-1f128).powi(l as i32) * fact(self.nu - l - 1)
                    / fact(l)
                    / fact(k - l)
                    / fact(k - l + self.nu);
            }
            res += temp * (z / 2.).powi(2 * k as i32 - 1) * k as f128 / 2.;
        }
        for k in 0..(n + 1).max(self.nu) - self.nu {
            let mut temp = 0.;
            let mut temp1 = 0.;
            let mut sum_harmonic = self.sum_harmonic_init;
            for l in 0..k + 1 {
                let f = 1. / fact(l) / fact(l + self.nu) / fact(k - l) / fact(k - l + self.nu);
                temp += (sum_harmonic / 2. - EULER_GAMMA - (z / 2.).ln()) * f;
                temp1 += (-1. / z) * f;
                sum_harmonic += 1. / (self.nu + l + 1) as f128 + 1. / (l + 1) as f128;
            }
            res += temp
                * (z / 2.).powi(2 * (k + self.nu) as i32 - 1)
                * (k + self.nu) as f128
                * (-1f128).powi(self.nu as i32)
                + temp1 * (z / 2.).powi(2 * (k + self.nu) as i32) * (-1f128).powi(self.nu as i32);
        }
        res
    }
    fn high_z_first_deriv(&self, z: f128, n: usize) -> f128 {
        let p = (1. + (z / self.nu as f128).powi(2)).powf(-0.5);
        let mut init = HashMap::new();
        init.insert(0, 1f128);
        let mut coefs = vec![init];

        let mut res = 0.;
        for k in 1..2 * n + 1 {
            let mut new_coef = HashMap::new();
            for (&i, c) in &coefs[k - 1] {
                let entry = new_coef.entry(i + 1).or_insert(0.);
                *entry += c * (2. * i as f128 + 1.).powi(2) / 8. / (i as f128 + 1.);
                let entry = new_coef.entry(i + 3).or_insert(0.);
                *entry -= c * (i as f128 / 2. + 5. / 8. / (3. + i as f128));
            }
            coefs.push(new_coef);
        }
        for k in 0..(n + 1) {
            let mut temp = 0.;
            for m in 0..2 * k + 1 {
                let u_1 = coefs[m]
                    .iter()
                    .map(|(&i, c)| c * p.powi(i))
                    .fold(0., |acc, x| acc + x);
                let u_2 = coefs[2 * k - m]
                    .iter()
                    .map(|(&i, c)| c * p.powi(i))
                    .fold(0., |acc, x| acc + x);
                let du_1 = coefs[m]
                    .iter()
                    .map(|(&i, c)| c * p.powi(i - 1) * i as f128)
                    .fold(0., |acc, x| acc + x);
                let du_2 = coefs[2 * k - m]
                    .iter()
                    .map(|(&i, c)| c * p.powi(i - 1) * i as f128)
                    .fold(0., |acc, x| acc + x);
                temp += (-1f128).powi(m as i32) * (u_1 * u_2 + p * u_1 * du_2 + p * du_1 * u_2);
            }
            res += temp / (self.nu as f128).powi(2 * k as i32);
        }
        -z * res * p.powi(3) / 2. / self.nu.pow(3) as f128
    }
}
