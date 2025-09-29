use rand::Rng;

pub enum DistributionType {
    Discrete,
    Continuous,
}

/// Sample direction isotropically
/// # Example
/// ```
/// ```
pub fn sample_direction_isotropically() -> [f64; 3] {
    let mut rng = rand::thread_rng();

    // sample \theta
    let _xsi_cos_theta: f64 = rng.r#gen();
    let _cos_theta = -1. + 2. * _xsi_cos_theta;

    // sample \phi
    let _xsi_phi: f64 = rng.r#gen();
    let _phi = 2. * std::f64::consts::PI * _xsi_phi;

    // return direction
    [
        (1. - _cos_theta * _cos_theta).sqrt() * _phi.cos(),
        (1. - _cos_theta * _cos_theta).sqrt() * _phi.sin(),
        _cos_theta,
    ]
}
