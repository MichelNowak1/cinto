use crate::common::geometry::Geometry;
use crate::common::cross_section_library::Interaction;
use crate::common::cross_section_library::CrossSectionLibrary;
use crate::monte_carlo::tally::Tally;
use crate::monte_carlo::trajectory::Trajectory;
use pyo3::prelude::*;

#[pyclass]
pub struct Detector {
    pub volume_name: String,
    pub response_type: Interaction,
    pub tally: Tally,
    pub importance: f64,
    pub scoring: bool,
    pub energy_bounds: Vec<f64>,
}

#[pymethods]
impl Detector {
    #[new]
    pub fn new(
        volume_name: String,
        interaction: String,
        energy_bounds: Vec<f64>) -> Self{
        let int = if interaction == "elastic_scattering" {
            Interaction::ElasticScattering
        } else {
            Interaction::Null
        };

        Self{
            volume_name: volume_name,
            response_type: int,
            energy_bounds: energy_bounds,
            tally: Tally::new(),
            importance: std::f64::MAX,
            scoring: true,
        }
    }
    pub fn score(
        &mut self,
        trajectory: &Trajectory,
        geometry: &Geometry,
        cross_section_library: &CrossSectionLibrary,
    ) {
        for pt in &trajectory.branches[0].lock().unwrap().points {
            let mut point = pt.lock().unwrap();
            if geometry.get_volume_name(point.position) == self.volume_name {
                point.importance = std::f64::MAX;

                let material = geometry.get_material(point.position);

                let sigma_tot = material.get_sigma(
                    Interaction::Total,
                    point.energy,
                    &cross_section_library);

                let score = point.weight / sigma_tot;

                point.contribution += score;
                self.add_score(score, point.importance);
            }
        }
    }
}

impl Detector {

    pub fn add_score(
        &mut self,
        score: f64,
        new_importance: f64) {
        if self.scoring {
            if new_importance < self.importance {
                self.importance = new_importance;
            }
            self.tally.add_value(score);
        }
    }
    pub fn prepare_next_batch(&mut self) {
        self.importance = std::f64::MAX;
        self.scoring = true;
        self.tally.prepare_next_batch();
    }
    pub fn get_current_score(&self) -> f64 {
        self.tally.get_current_value()
    }
    pub fn get_mean(&self) -> f64 {
        self.tally.current_value;
        self.tally.get_mean()
    }
    pub fn get_standard_deviation(&self) -> f64 {
        self.tally.get_standard_deviation()
    }
    pub fn normalize_ams_weight(&mut self, ams_weight: f64) {
        self.tally.current_value *= ams_weight;
    }
    pub fn normalize(&mut self, num_particles_per_batch: f64) {
        self.tally.current_value /= num_particles_per_batch;
    }
}
