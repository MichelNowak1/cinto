use crate::common::cross_section_library::CrossSectionLibrary;
use crate::common::cross_section_library::EvaluationType;
use crate::common::cross_section_library::Interaction;
use crate::common::isotope::Isotope;
use crate::common::utils;

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::sync::Arc;

static AVOGADRO_SCALE: f64 = 0.6022094;

/// Material
///
/// # Attributes
/// * `density` density of the material in the geometry $(g/cm^3)$
/// * `isotope_mcnp_id` mcnp id of isotopes
/// * `isotope_concentrations` concentrations of the isotopes in the material
#[pyclass]
#[derive(Clone)]
pub struct Material {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub density: f64,
    #[pyo3(get, set)]
    pub isotope_names: Vec<String>,
    #[pyo3(get, set)]
    pub isotope_concentrations: Vec<f64>,
}

#[pymethods]
impl Material {
    #[new]
    pub fn new(name: String) -> Self {
        println!("loading material: {}", name);

        Python::with_gil(|py| {
            let cinto_pyne = py.import("cinto").unwrap();

            let py_material_name = PyTuple::new(py, &[&name]);
            let py_material = cinto_pyne
                .getattr("get_material")
                .unwrap()
                .call1(py_material_name)
                .unwrap()
                .extract::<PyObject>()
                .unwrap();

            // get number of isotopes in material
            let isotope_names: Vec<String> = py_material
                .getattr(py, "isotope_names")
                .unwrap()
                .extract(py)
                .unwrap();

            // get density
            let density: f64 = py_material
                .getattr(py, "density")
                .unwrap()
                .extract(py)
                .unwrap();

            // get concentrations of isotopes in material
            let concentrations: Vec<f64> = py_material
                .getattr(py, "concentrations")
                .unwrap()
                .extract(py)
                .unwrap();

            println!("density: {}", density);
            let mat = Material {
                name: name,
                density: density,
                isotope_names: isotope_names,
                isotope_concentrations: concentrations,
            };

            return mat;
        })
    }
}
impl Material {
    /// Sample an isotope to interact with
    ///
    /// # Arguments
    /// * `energy` energy at which the isotope needs to be sampled
    /// * `cross_section_library` the library that provides with the cross sections
    ///
    /// # Returns
    /// * `isotope` a reference to the isotope sampled
    ///
    /// # Example
    ///
    /// ```
    /// let energy = 0.2; // (MeV)
    /// let sampled_isotope = material.sample_isotope(energy, cross_section_library);
    /// ```
    pub fn sample_isotope(
        &self,
        energy: f64,
        cross_section_library: &CrossSectionLibrary,
    ) -> Arc<Isotope> {
        if self.isotope_names.len() == 1 {
            return cross_section_library.get_isotope(&self.isotope_names[0]);
        }
        let mut probability_distribution = vec![0.; self.isotope_names.len()];

        let mut x = vec![0.; self.isotope_names.len() + 1];

        for (i, isotope_name) in self.isotope_names.iter().enumerate() {
            x[i + 1] = i as f64;
            let isotope = cross_section_library.get_isotope(&isotope_name);
            let sigma_tot = isotope.get_sigma_tot(energy);
            let concentration = self.isotope_concentrations[i];
            probability_distribution[i] = sigma_tot * concentration;
        }

        let isotope_index = utils::sample_from_distribution(
            &x,
            &probability_distribution) as usize;

        if isotope_index >= self.isotope_names.len() {
            panic!("Isotope not found when sampling.")
        }
        cross_section_library.get_isotope(&self.isotope_names[isotope_index])
    }

    /// Get material sigma abs
    ///
    /// # Arguments
    /// * `energy` energy at which the macroscopic total cross sections needs to be retrieved
    /// * `interaction` interaction for which the cross section needs to be retrieved
    /// * `cross_section_library` the library that provides with the cross sections
    ///
    /// # Returns
    /// * `sigma` macroscopic absorption cross section at energy `energy`
    ///
    /// # Example
    /// ```
    /// let energy = 0.4; // (MeV)
    /// let sigma_tot = material.get_sigma(Interaction::Total, energy, cross_section_library);
    /// ```
    pub fn get_sigma(
        &self,
        interaction: Interaction,
        energy: f64,
        cross_section_library: &CrossSectionLibrary,
    ) -> f64 {

        let mut sigma = 0.;

        for (isotope_name, concentration) in 
            self.isotope_names
            .iter()
            .zip(self.isotope_concentrations.iter())
        {
            let isotope = cross_section_library.get_isotope(&isotope_name);

            let micro_sigma = match interaction {
                Interaction::Total => isotope.get_sigma_tot(energy),
                Interaction::ElasticScattering => isotope.get_sigma_s(energy),
                Interaction::Absorption => isotope.get_sigma_abs(energy),
                _ => panic!("not implemented {:?}", interaction),
            };

            if cross_section_library.evaluation_type == EvaluationType::Punctual {
                sigma += self.density *
                         concentration *
                         AVOGADRO_SCALE /
                         isotope.atomic_mass
                       * micro_sigma;
            } else if cross_section_library.evaluation_type == EvaluationType::Multigroup {
                // already multigroup cross section
                sigma += micro_sigma;
            }
        }
        sigma
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_material_new() {
        let _material = Material::new("Water, Liquid".to_string());
    }
}
