pub trait Potential {
    // fn val(&self, phi: f128) -> f128;
    fn first_deriv(&self, phi: f128) -> f128;
    fn second_deriv(&self, phi: f128) -> f128;
    fn third_deriv(&self, phi: f128) -> f128;
    fn forth_deriv(&self, phi: f128) -> f128;
    fn phi_fv(&self) -> f128;
    fn phi_tv(&self) -> f128;
    fn phi_top(&self) -> f128;
    fn second_deriv_fv(&self) -> f128;
}
