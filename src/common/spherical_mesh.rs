use crate::common::geometry::Geometry;
use crate::common::cross_section_library::CrossSectionLibrary;
use crate::common::mesh::Mesh;
use crate::common::utils;
use crate::monte_carlo::trajectory::Point;
use crate::monte_carlo::trajectory::Trajectory;
use crate::monte_carlo::tally;
use ndarray::prelude::*;

pub struct SphericalMesh {
    pub center: [f64; 3],
    pub r_bounds: Vec<f64>,
    pub theta_bounds: Vec<f64>,
    pub phi_bounds: Vec<f64>,
    pub energy_bounds: Vec<f64>,
    pub tallies: Array4<tally::Tally>,
}

impl Mesh for SphericalMesh {
    fn get_index(&self, particle: &Point) -> Option<[usize; 4]> {
        let x = particle.position[0] - self.center[0];
        let y = particle.position[1] - self.center[1];
        let z = particle.position[2] - self.center[2];

        let r: f64 = (x.powi(2) + y.powi(2) + z.powi(2)).sqrt();
        let theta: f64 = ((y / x).atan() + std::f64::consts::PI / 2.) % std::f64::consts::PI;
        let phi: f64 = (((x.powi(2) + y.powi(2)).sqrt() / (z.powi(2))).atan() * 2.)
            % (2. * std::f64::consts::PI);

        let r_index = utils::get_index(r, &self.r_bounds);
        let theta_index = utils::get_index(theta, &self.theta_bounds);
        let phi_index = utils::get_index(phi, &self.phi_bounds);
        let energy_index = utils::get_index(particle.energy, &self.energy_bounds);

        let index = [r_index, theta_index, phi_index, energy_index];
        for id in index {
            if !id.is_some() {
                return None;
            }
        }
        Some([
            r_index.unwrap(),
            theta_index.unwrap(),
            phi_index.unwrap(),
            energy_index.unwrap(),
        ])
    }
    fn add_tally(
        &mut self,
        trajectory: &Trajectory,
        geometry: &Geometry,
        cross_section_library: &CrossSectionLibrary,
    ) {
        panic!("Not implemented yet")
    }
    fn normalize(&mut self, norm: f64) {
        for tally in self.tallies.iter_mut() {
            tally.current_value /= norm;
        }
    }
    fn prepare_next_batch(&mut self) {
        for tally in self.tallies.iter_mut() {
            tally.prepare_next_batch();
        }
    }
    fn normalize_ams_weight(&mut self, ams_weight: f64) {
        for tally in self.tallies.iter_mut() {
            tally.current_value *= ams_weight;
        }
    }
    fn dump(&self) {
        let mut means = Vec::<f64>::new();
        let mut stds = Vec::<f64>::new();
        for tally in self.tallies.iter() {
            let mean = tally.get_mean();
            let std = tally.get_standard_deviation();
            means.push(mean);
            stds.push(std);
        }
        println!("{:?}", means);
        println!("");
    }
}
