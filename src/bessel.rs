use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;
    use dbgbb::dbgbb;
    use ndarray::Array1;
    #[test]
    fn compare() {
        let z: Array1<f128> = (0..100).map(|i| i as f128 / 100. * 50.).collect();
        let res = z.map(|&x| x.besselik(20) as f64);
        let z = z.map(|&x| x as f64);
        dbgbb!(z, res);
    }
}

use special_fun::FloatSpecial;

pub trait BesselIK {
    fn besselik(&self, nu: usize) -> Self;
    fn besselik_deriv(&self, nu: usize, n: usize) -> Self;
}

impl BesselIK for f128 {
    fn besselik(&self, nu: usize) -> f128 {
        let z = *self;
        ((z as f64).besseli(nu as f64) * (z as f64).besselk(nu as i32)) as f128
    }
    fn besselik_deriv(&self, nu: usize, n: usize) -> f128 {
        let z = *self;
        let mut coefs = HashMap::new();
        coefs.insert((nu, nu), 1f128);
        for _ in 0..n {
            let mut coefs_new = HashMap::new();
            for ((nu_i, nu_k), c) in coefs {
                let entry = coefs_new.entry((nu_i + 1, nu_k)).or_insert(0.);
                *entry += 0.5 * c;
                let entry = coefs_new
                    .entry((if nu_i == 0 { 1 } else { nu_i - 1 }, nu_k))
                    .or_insert(0.);
                *entry += 0.5 * c;
                let entry = coefs_new.entry((nu_i, nu_k + 1)).or_insert(0.);
                *entry += -0.5 * c;
                let entry = coefs_new
                    .entry((nu_i, if nu_k == 0 { 1 } else { nu_k - 1 }))
                    .or_insert(0.);
                *entry += -0.5 * c;
            }
            coefs = coefs_new;
        }
        coefs
            .iter()
            .map(|((nu_i, nu_k), c)| {
                *c as f64 * (z as f64).besseli(*nu_i as f64) * (z as f64).besselk(*nu_k as i32)
            })
            .sum::<f64>() as f128
    }
}
