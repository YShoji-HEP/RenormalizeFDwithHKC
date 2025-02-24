use russell_lab::math;
use std::collections::HashMap;
#[cfg(test)]
mod tests {
    use super::*;
    use dbgbb::dbgbb;
    use ndarray::Array1;
    #[test]
    fn compare() {
        let zmax = 0.2;
        let z: Array1<f128> = (0..100).map(|i| i as f128 / 100. * zmax).collect();
        let bessel = z.map(|&x| x.besselik_deriv(5, 5) as f64);
        let z = z.map(|&x| x as f64);
        dbgbb!(z, bessel);
    }
}

pub trait BesselIK {
    fn besselik(&self, nu: usize) -> Self;
    fn besselik_deriv(&self, nu: usize, n: usize) -> Self;
}

fn low_z(nu: usize, z: f128, p: usize) -> f128 {
    let fact = |i: usize| ((i + 1) as f128).gamma();
    let mut res = 0.;
    for k in 0..p + 1 {
        let mut temp = 0.;
        for l in 0..(k + 1).min(nu) {
            temp += (-1f128).powi(l as i32) * fact(nu - l - 1)
                / fact(l)
                / fact(k - l)
                / fact(k - l + nu);
        }
        res += temp * (z / 2.).powi(2 * k as i32) / 2.;
    }
    res
}

fn low_z_deriv(nu: usize, z: f128, p: usize, n: usize) -> f128 {
    let fact = |i: usize| ((i + 1) as f128).gamma();
    let mut res = 0.;
    for k in 0..p + 1 {
        if 2 * k + 1 > n {
            let mut temp = 0.;
            for l in 0..(k + 1).min(nu) {
                temp += (-1f128).powi(l as i32) * fact(nu - l - 1)
                    / fact(l)
                    / fact(k - l)
                    / fact(k - l + nu);
            }
            res += temp * (z / 2.).powi(2 * k as i32 - n as i32) * fact(2 * k)
                / fact(2 * k - n)
                / 2f128.powi(n as i32 + 1);
        }
    }
    res
}

impl BesselIK for f128 {
    fn besselik(&self, nu: usize) -> f128 {
        let z = *self;
        if nu > 7 && z < 0.1 {
            low_z(nu, z, 7)
        } else {
            (math::bessel_in(nu, z as f64) * math::bessel_kn(nu.try_into().unwrap(), z as f64))
                as f128
        }
    }
    fn besselik_deriv(&self, nu: usize, n: usize) -> f128 {
        let z = *self;
        if nu > 7 && z < 0.1 {
            low_z_deriv(nu, z, 7, n)
        } else {
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
                    *c as f64
                        * math::bessel_in(*nu_i, z as f64)
                        * math::bessel_kn((*nu_k).try_into().unwrap(), z as f64)
                })
                .sum::<f64>() as f128
        }
    }
}
