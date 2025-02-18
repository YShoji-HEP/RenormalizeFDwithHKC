
pub trait Potential {
    fn val(&self, psi: &f64) -> f64;
    fn first_deriv(&self, psi: &f64) -> f64;
    fn second_deriv(&self, psi: &f64) -> f64;
    fn phi_fv(&self)->f64;
    fn phi_tv(&self)->f64;
    fn phi_top(&self)->f64;
}
