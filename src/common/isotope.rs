use crate::common::cross_section::CrossSection;
use crate::common::cross_section_library::Interaction;
use crate::common::utils;

use ndarray::prelude::*;
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct Isotope {
    pub name: String,
    pub atomic_mass: f64,
    pub energies: Vec<f64>,
    pub is_fissile: bool,
    pub cross_sections: HashMap<Interaction, Box<dyn CrossSection + Send + Sync>>,
    pub scattering_matrix: Option<Array2<f64>>,
}

impl Isotope {
    pub fn sample_interaction(&self, energy: f64) -> Interaction {
        let mut probability_distribution = Vec::<f64>::new();
        let mut possible_interactions = Vec::<Interaction>::new();

        for (interaction, cross_section) in &self.cross_sections {
            if interaction != &Interaction::Total {
                if cross_section.get_value(
                    energy,
                    &self.energies) == 0. {
                    continue;
                }
                probability_distribution.push(
                    cross_section.get_value(
                        energy,
                        &self.energies));
                possible_interactions.push(interaction.clone());
            }
        }

        let interaction = utils::sample_from_discrete_distribution(
            &possible_interactions,
            probability_distribution,
        );
        interaction
    }

    pub fn get_energies(&self) -> &Vec<f64> {
        &self.energies
    }

    pub fn get_sigma_tot(&self, energy: f64) -> f64 {
        let cross_sections = self.cross_sections.get(&Interaction::Total);

        if cross_sections.is_some() {
            cross_sections.unwrap().get_value(energy, &self.energies)
        } else {
            0.
        }
    }

    pub fn get_sigma_s(&self, energy: f64) -> f64 {
        let cross_sections = self.cross_sections.get(&Interaction::ElasticScattering);

        if cross_sections.is_some() {
            cross_sections.unwrap().get_value(energy, &self.energies)
        } else {
            0.
        }
    }

    pub fn get_sigma_abs(&self, energy: f64) -> f64 {
        if self.is_fissile {
            self.get_sigma_tot(energy) - self.get_sigma_s(energy)
        } else {
            self.get_sigma_tot(energy) - self.get_sigma_s(energy)
        }
    }
}
