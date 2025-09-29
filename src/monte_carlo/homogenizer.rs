use ndarray::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::common::cross_section::CrossSection;
use crate::common::cross_section::MultiGroupCrossSection;
use crate::common::cross_section_library::CrossSectionLibrary;
use crate::common::cross_section_library::Interaction;
use crate::common::geometry::Geometry;
use crate::common::isotope::Isotope;
use crate::common::utils;

use crate::monte_carlo::particle::Particle;
use crate::monte_carlo::tally::Tally;

pub trait Homogenizer {
    fn is_some(&self) -> bool;
    fn init(&mut self, geometry: &Geometry);
    fn score(
        &mut self,
        particle: &Particle,
        geometry: &Geometry,
        cross_section_library: &CrossSectionLibrary,
    );
    fn score_scattering_matrix(
        &mut self,
        particle: &Particle,
        geometry: &Geometry,
        cross_section_library: &CrossSectionLibrary,
    );
    fn normalize(&mut self, source_norm: f64);
    fn prepare_next_batch(&mut self);
    fn update_cross_section_library(&mut self);
    fn write_cross_section_library_to_file(&self);
}

/// Null dummy Homogenizer
pub struct Null {}
impl Homogenizer for Null {
    fn init(&mut self, _geometry: &Geometry) {}
    fn score(
        &mut self,
        _particle: &Particle,
        _geometry: &Geometry,
        _cross_section_library: &CrossSectionLibrary,
    ) {
    }
    fn score_scattering_matrix(
        &mut self,
        _particle: &Particle,
        _geometry: &Geometry,
        _cross_section_library: &CrossSectionLibrary,
    ) {
    }
    fn normalize(&mut self, _source_norm: f64){}
    fn prepare_next_batch(&mut self) {}
    fn is_some(&self) -> bool {
        return false;
    }
    fn update_cross_section_library(&mut self) {}
    fn write_cross_section_library_to_file(&self) {}
}

/// VolumeHomogenizer
///
/// # Attributes
/// * `cross_section_library` cross section library from which to homogenize the cross sections
/// * `energy_bounds` energy bounds of to be taken into account for the multigroup discretization
/// * `flux_tallies` tallies representing the flux for each volume
/// * `total_xs_tallies` tallies representing the total cross sections for each volume
/// * `scattering_xs_tallies` tallies representing the scattering cross sections for each volume
/// * `scattering_matrix_tallies` tallies representing the scattering matrix for each volume
/// * `to_file` name of the file to which the homogenized cross section library needs to be saved
pub struct VolumeHomogenizer {
    pub cross_section_library: CrossSectionLibrary,
    pub energy_bounds: Vec<f64>,
    pub flux_tallies: HashMap<String, Vec<Tally>>,
    pub total_xs_tallies: HashMap<String, Vec<Tally>>,
    pub scattering_xs_tallies: HashMap<String, Vec<Tally>>,
    pub scattering_matrix_tallies: HashMap<String, Box<Array2<Tally>>>,
    pub to_file: Option<String>,
}

impl Homogenizer for VolumeHomogenizer {
    /// Initializes the VolumeHomogenizer
    ///
    /// # Arguments
    /// * `geometry` geometry of the problem being solved
    fn init(&mut self, geometry: &Geometry) {
        let num_energy_groups = self.energy_bounds.len() - 1;
        for volume_name in geometry.volume_material_association.keys() {
            self.total_xs_tallies.insert(
                volume_name.to_string(),
                vec![Tally::new(); num_energy_groups],
            );
            self.scattering_xs_tallies.insert(
                volume_name.to_string(),
                vec![Tally::new(); num_energy_groups],
            );
            self.flux_tallies.insert(
                volume_name.to_string(),
                vec![Tally::new(); num_energy_groups],
            );
            self.scattering_matrix_tallies.insert(
                volume_name.to_string(),
                Box::new(Array2::<Tally>::zeros((
                    num_energy_groups,
                    num_energy_groups,
                ))),
            );
        }
    }

    /// Scored cross section tallies
    ///
    /// # Arguments
    /// * `particle` particle that contributes to the homogenized cross sections tallies
    /// * `geometry` geometry of the problem being solved
    /// * `cross_section_library` cross section library from which to homogenize the cross sections
    fn score(
        &mut self,
        particle: &Particle,
        geometry: &Geometry,
        cross_section_library: &CrossSectionLibrary,
    ) {
        let volume_name = geometry.get_volume_name(
            particle.position);

        // initial energy, before collision
        let energy_index_ = utils::get_index(
            particle.energy,
            &self.energy_bounds);

        // do not score if energy does not correspond to energetic mesh
        if !energy_index_.is_some() {
            return;
        }
        let energy_index = energy_index_.unwrap();
        let material = geometry.get_material(particle.position);

        let total_xs = material.get_sigma(
            Interaction::Total,
            particle.energy,
            &cross_section_library);

        let scattering_xs = material.get_sigma(
            Interaction::ElasticScattering,
            particle.energy,
            &cross_section_library,
        );

        self.total_xs_tallies.get_mut(&volume_name)
                             .unwrap()[energy_index]
                             .add_value(particle.weight);

        self.scattering_xs_tallies.get_mut(&volume_name)
                                  .unwrap()[energy_index]
                                  .add_value(particle.weight *
                                             scattering_xs /
                                             total_xs);
        self.flux_tallies.get_mut(&volume_name)
                         .unwrap()[energy_index]
                         .add_value(particle.weight / total_xs);
    }

    /// Scored scattering matrix tallies
    ///
    /// # Arguments
    /// * `particle` particle that contributes to the homogenized cross sections tallies
    /// * `geometry` geometry of the problem being solved
    /// * `cross_section_library` cross section library from which to homogenize the cross sections
    fn score_scattering_matrix(
        &mut self,
        particle: &Particle,
        geometry: &Geometry,
        cross_section_library: &CrossSectionLibrary,
    ) {
        if particle.last_interaction != Interaction::ElasticScattering {
            return;
        }
        let volume_name = geometry.get_volume_name(particle.position);
        let material = geometry.get_material(particle.position);

        let scattering_source = utils::get_index(
            particle.initial_energy,
            &self.energy_bounds);

        let scattering_destination = utils::get_index(
            particle.energy,
            &self.energy_bounds);

        if !scattering_source.is_some() || !scattering_destination.is_some() {
            return;
        }

        let scattering_xs = material.get_sigma(
            Interaction::ElasticScattering,
            particle.initial_energy,
            &cross_section_library,
        );

        let total_xs = material.get_sigma(
            Interaction::Total,
            particle.initial_energy,
            &cross_section_library,
        );

        self.scattering_matrix_tallies.get_mut(&volume_name)
            .unwrap()[( scattering_destination.unwrap(), scattering_source.unwrap())]
            .add_value(scattering_xs * particle.weight / total_xs);
    }
    /// Tells if homoegenizer is not Null
    ///
    /// # Returns
    /// * `is_some` true
    fn is_some(&self) -> bool {
        return true;
    }

    fn normalize(&mut self, norm: f64){
        for (key, tallies) in &mut self.total_xs_tallies{
            for tally in tallies.iter_mut(){
                tally.current_value /= norm;
            }
        }
        for (key, tallies) in &mut self.scattering_xs_tallies{
            for tally in tallies.iter_mut(){
                tally.current_value /= norm;
            }
        }
        for (key, tallies) in &mut self.flux_tallies{
            for tally in tallies.iter_mut(){
                tally.current_value /= norm;
            }
        }
        for (key, tallies) in &mut self.scattering_matrix_tallies{
            for col in 0..tallies.ncols(){
                for row in 0..tallies.nrows(){
                    tallies[(row, col)].current_value /= norm;
                }
            }
        }
    }
    /// Prepares next batch by preparing all tallies used to homogenize the cross section library
    fn prepare_next_batch(&mut self) {
        for (_, tallies) in &mut self.total_xs_tallies {
            for tally in tallies.iter_mut() {
                tally.prepare_next_batch();
            }
        }
        for (_, tallies) in &mut self.scattering_xs_tallies {
            for tally in tallies.iter_mut() {
                tally.prepare_next_batch();
            }
        }
        for (_, tallies) in &mut self.flux_tallies {
            for tally in tallies.iter_mut() {
                tally.prepare_next_batch();
            }
        }
        for (_, tallies) in &mut self.scattering_matrix_tallies {
            for tally in tallies.iter_mut() {
                tally.prepare_next_batch();
            }
        }
    }

    /// Updates the cross section library with current tallies
    fn update_cross_section_library(&mut self) {
        self.cross_section_library.isotopes.clear();
        for volume_name in self.total_xs_tallies.keys() {
            let mut cross_sections =
                HashMap::<Interaction, Box<dyn CrossSection + Send + Sync>>::new();

            let mut values = Vec::new();

            // total xs
            for (total_tally, flux_tally) in self
                .total_xs_tallies
                .get(volume_name)
                .unwrap()
                .iter()
                .zip(self.flux_tallies.get(volume_name).unwrap().iter())
            {
                values.push(total_tally.get_mean() / flux_tally.get_mean());
            }
            let total_xs = MultiGroupCrossSection {
                values: values.clone(),
            };
            cross_sections.insert(Interaction::Total, Box::new(total_xs));

            values.clear();

            for (scattering_tally, flux_tally) in self
                .scattering_xs_tallies
                .get(volume_name)
                .unwrap()
                .iter()
                .zip(self.flux_tallies.get(volume_name).unwrap().iter())
            {
                values.push(
                    scattering_tally.get_mean() /
                    flux_tally.get_mean());
            }
            let scattering_xs = MultiGroupCrossSection {
                values: values.clone(),
            };
            cross_sections.insert(Interaction::ElasticScattering, Box::new(scattering_xs));

            values.clear();

            // scattering matrix
            let num_energy_groups = self.energy_bounds.len() - 1;
            let mut scattering_matrix =
                Array2::<f64>::zeros((num_energy_groups, num_energy_groups));
            for e1 in 0..num_energy_groups {
                for e2 in 0..num_energy_groups {
                    let scattering_tally_mean =
                        self.scattering_matrix_tallies.get(volume_name)
                                                      .unwrap()[(e1, e2)]
                                                      .get_mean();
                    let flux_tally_mean =
                        self.flux_tallies.get(volume_name)
                                         .unwrap()[e2]
                                         .get_mean();

                    scattering_matrix[(e1, e2)] = scattering_tally_mean / flux_tally_mean;
                }
            }

            let isotope = Isotope {
                name: volume_name.to_string(),
                energies: self.energy_bounds.clone(),
                atomic_mass: 1.,
                is_fissile: false,
                cross_sections: cross_sections,
                scattering_matrix: Some(scattering_matrix),
            };
            self.cross_section_library
                .isotopes
                .insert(volume_name.to_string(), Arc::new(isotope));
        }
    }

    /// Writes cross section library into file with file name to_file
    fn write_cross_section_library_to_file(&self) {
        if self.to_file.is_some() {
            self.cross_section_library
                .save_to_file(self.to_file.as_ref()
                                          .unwrap()
                                          .to_string())
                .unwrap();
        }
    }
}
