extern crate rand;
use crate::common::cross_section_library::Interaction;
use crate::monte_carlo::particle;
use crate::monte_carlo::utils;

use rand::Rng;
use pyo3::prelude::*;

/// Source
/// Punctual source
///
/// # Arguments
/// * `position` position of the punctual source
/// * `direction` direction of the punctual source
/// * `energy` energy of the punctual source

#[pyclass]
pub struct Source {
    pub position: [f64; 3],
    pub direction: [f64; 3],
    pub energy: f64,
}

#[pymethods]
impl Source {
    #[new]
    pub fn new(position: [f64; 3], direction: [f64; 3], energy: f64) -> Self{
        Self {
            position: position,
            direction: direction,
            energy: energy,
        }
    }
    pub fn get_position(&self) -> [f64; 3] {
        let mut rng = rand::thread_rng();
        let _xsi: f64 = rng.r#gen();

        self.position
    }

    pub fn get_direction(&self) -> [f64; 3] {
        if self.direction == [0., 0., 0.] {
            let _direction = utils::sample_direction_isotropically();
            _direction
        } else {
            self.direction
        }
    }

    pub fn get_energy(&self) -> f64 {
        self.energy
    }

    /// Sample a particle
    ///
    /// # Returns
    /// * `particle` particle sampled according to the punctual source parameters
    ///
    /// # Example
    /// ```
    /// let source_position = [0., 0., 0.];
    /// let source_direction = [1., 0., 0.];
    /// let source_energy = 2; // MeV
    ///
    /// let source = PunctualSource{
    ///     position: source_position,
    ///     direction: source_direction,
    ///     energy: source_energy ,
    /// };
    ///
    /// let source_particle = source.sample();
    /// ```
    pub fn sample(&self) -> particle::Particle {
        particle::Particle {
            position: self.get_position(),
            direction: self.get_direction(),
            energy: self.get_energy(),
            time: 0.,
            weight: 1.,
            importance: -1.,
            contribution: 0.,
            is_colliding: false,
            is_absorbed: false,
            is_crossing: false,
            mother_trajectory_id: -1,
            mother_branch_id: -1,
            rank_in_mother_branch: -1,
            last_interaction: Interaction::Source,
            initial_energy: self.get_energy(),
        }
    }
}
