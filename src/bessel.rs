use std::collections::HashMap;

struct BesselIK {
    nu: usize,
    sum_harmonic_init: f128,
}

const EULER_GAMMA: f128 = 0.577215664901532860606512090082;

impl BesselIK {
    fn new(nu: usize) -> Self {
        let sum_harmonic_init: f128 = (1..nu + 1)
            .map(|i| 1. / i as f128)
            .fold(0., |acc, x| acc + x);
        Self {
            nu,
            sum_harmonic_init,
        }
    }
    fn low_z_int(&self, z: f128, n: usize) -> f128 {
        let fact = |i: usize| ((i + 1) as f128).gamma();
        let mut res = 0.;
        for k in 0..n {
            let mut temp = 0.;
            for l in 0..k.min(self.nu.max(1) - 1) {
                temp += (-1f128).powi(l as i32) * fact(self.nu - l - 1)
                    / fact(l)
                    / fact(k - l)
                    / fact(k - l + self.nu);
            }
            res += temp * (z / 2.).powi(2 * k as i32) / 2.;
        }
        for k in 0..n.max(self.nu) - self.nu {
            let mut temp = 0.;
            let mut sum_harmonic = self.sum_harmonic_init;
            for l in 0..k.min(self.nu.max(1) - 1) {
                temp += (sum_harmonic / 2. - EULER_GAMMA - (z / 2.).ln())
                    / fact(l)
                    / fact(l + self.nu)
                    / fact(k - l)
                    / fact(k - l + self.nu);
                sum_harmonic += 1. / (self.nu + l + 1) as f128 + 1. / (l + 1) as f128;
            }
            res += temp * (z / 2.).powi(2 * (k + self.nu) as i32);
        }
        res
    }
    fn high_z(&self, z: f128, n: usize) -> f128 {
        let p = (1. + (z / self.nu as f128).powi(2)).powf(-0.5);
        let mut init = HashMap::new();
        init.insert(0, 0f128);
        let mut coefs = vec![init];

        let mut res = 0.;
        for k in 0..n {
            for j in 2 * k.max(1) - 1..2 * k + 1 {
                let mut new_coef = HashMap::new();
                for (&n, c) in &coefs[j] {
                    let entry = new_coef.entry(n + 1).or_insert(0.);
                    *entry += c * (2. * n as f128 + 1.).powi(2) / 8. / (n as f128 + 1.);
                    let entry = new_coef.entry(n + 3).or_insert(0.);
                    *entry -= c * (n as f128 / 2. + 5. / 8. / (3. + n as f128));
                }
                coefs.push(new_coef);
            }
            let mut temp = 0.;
            for m in 0..2 * k + 1 {
                let u_1 = coefs[m]
                    .iter()
                    .map(|(&n, c)| c * z.powi(n))
                    .fold(0., |acc, x| acc + x);
                let u_2 = coefs[2 * k - m]
                    .iter()
                    .map(|(&n, c)| c * z.powi(n))
                    .fold(0., |acc, x| acc + x);
                temp += (-1f128).powi(m as i32) * u_1 * u_2;
            }
            res += temp / (self.nu as f128).powi(2 * k as i32);
        }
        res * p / 2. / self.nu as f128
    }
}
