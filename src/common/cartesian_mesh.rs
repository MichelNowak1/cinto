use crate::common::cross_section_library::CrossSectionLibrary;
use crate::common::cross_section_library::Interaction;
use crate::common::geometry::Geometry;
use crate::common::mesh::Mesh;
use crate::common::utils;
use crate::monte_carlo::trajectory::Trajectory;
use crate::monte_carlo::trajectory::Point;
use crate::monte_carlo::tally;
use ndarray::prelude::*;

use std::fs::File;
use std::io::Write;

pub struct CartesianMesh {
    pub x_bounds: Vec<f64>,
    pub y_bounds: Vec<f64>,
    pub z_bounds: Vec<f64>,
    pub energy_bounds: Vec<f64>,

    pub tallies: Array4<tally::Tally>,
}

impl Mesh for CartesianMesh {
    fn get_index(&self, point: &Point) -> Option<[usize; 4]> {
        let x_index = utils::get_index(point.position[0], &self.x_bounds);
        let y_index = utils::get_index(point.position[1], &self.y_bounds);
        let z_index = utils::get_index(point.position[2], &self.z_bounds);
        let energy_index = utils::get_index(point.energy, &self.energy_bounds);

        let index = [x_index, y_index, z_index, energy_index];
        for id in index {
            if !id.is_some() {
                return None;
            }
        }
        Some([
            x_index.unwrap(),
            y_index.unwrap(),
            z_index.unwrap(),
            energy_index.unwrap(),
        ])
    }
    fn add_tally(
        &mut self,
        trajectory: &Trajectory,
        geometry: &Geometry,
        cross_section_library: &CrossSectionLibrary,
    ) {
        for pt in &trajectory.branches[0].lock().unwrap().points {
            let mut point = pt.lock().unwrap();

            let index_ = self.get_index(&point);
            if !index_.is_some() {
                return;
            }
            point.importance = std::f64::MAX;

            let material = geometry.get_material(point.position);

            let sigma_tot = material.get_sigma(
                Interaction::Total,
                point.energy,
                &cross_section_library);

            let score = point.weight / sigma_tot; // / sigma_tot;

            point.contribution += score;
            let index = index_.unwrap();

            let sigma_tot =
                material.get_sigma(
                    Interaction::Total,
                    point.energy,
                    &cross_section_library);

            let score = point.weight / sigma_tot;
            self.tallies[index].current_value += score;
        }
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
        let mut string = "".to_string();
        for mean in means.iter() {
            string += format!("{:e},", mean).as_str();
        }
        let mut mesh_file = File::create("mesh.txt").unwrap();
        writeln!(mesh_file, "{}", string).unwrap();
    }
}
