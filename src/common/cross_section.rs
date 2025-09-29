use crate::common::utils;

pub trait CrossSection {
    fn get_value(&self, energy: f64, energies: &Vec<f64>) -> f64;
    fn get_values(&self) -> &Vec<f64>;
}

/// Cross sections stored with the multigroup formalism
///
/// # Attributes
/// * `energy_bounds` energy bounds of the multigroup structure
/// * `values` values of the cross section in the energy groups
pub struct MultiGroupCrossSection {
    pub values: Vec<f64>,
}

impl CrossSection for MultiGroupCrossSection {
    /// retrieves the value of the cross section at a given energy
    ///
    /// # Attributes
    /// * `energy` energy at which the cross section needs to be found
    ///
    /// # Returns
    /// * `value` value of the cross section at given energy
    fn get_value(&self, energy: f64, energies: &Vec<f64>) -> f64 {
        let index = utils::get_index(energy, energies);
        if index.is_some() {
            self.values[index.unwrap()]
        } else {
            panic!("cross section asked at unsupported energy {}", energy);
        }
    }
    fn get_values(&self) -> &Vec<f64> {
        &self.values
    }
}
impl MultiGroupCrossSection {
    // pub fn build_from_detector(detector: Detector) {
    // }
}

/// Cross sections stored with the multigroup formalism
///
/// # Attributes
/// * `energies` energies at which the cross sections are evaluated
/// * `values` values of the cross sections
pub struct PunctualCrossSection {
    pub values: Vec<f64>,
}

impl CrossSection for PunctualCrossSection {
    /// retrieves the value of the cross section at a given energy
    ///
    /// # Attributes
    /// * `energy` energy at which the cross section needs to be found
    ///
    /// # Returns
    /// * `value` value of the cross section at given energy
    fn get_value(&self, energy: f64, energies: &Vec<f64>) -> f64 {
        utils::get_interpolated_value(energy, energies, &self.values)
    }
    fn get_values(&self) -> &Vec<f64> {
        &self.values
    }
}
