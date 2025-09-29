use itertools::iproduct;
use ndarray::prelude::*;
use serde_json::json;
use std::fs::File;
use std::io::Error;
use std::io::Write;

use crate::common::importance_map::importance::Importance;
use crate::common::utils;
use crate::monte_carlo::trajectory::Point;
use crate::monte_carlo::tally::Tally;
use crate::monte_carlo::trajectory;

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
#[pyclass]
pub struct DirectionalImportanceMap {
    pub x_bounds: Vec<f64>,
    pub y_bounds: Vec<f64>,
    pub z_bounds: Vec<f64>,
    pub energy_bounds: Vec<f64>,
    pub theta_bounds: Vec<f64>,
    pub phi_bounds: Vec<f64>,

    // importances
    pub importances: Array6<f64>,
    pub scored_importances_num: Array6<Tally>,
    pub scored_importances_denom: Array6<Tally>,
    pub num_visits: Array6<usize>,
}

impl Importance for DirectionalImportanceMap {

    fn get_index(&self, point: &trajectory::Point) -> Option<[usize; 6]> {
        let index: [i32; 6] = [0, 0, 0, 0, 0, 0];

        let vx = point.direction[0];
        let vy = point.direction[1];
        let vz = point.direction[2];

        // get theta and phi
        let theta: f64 = (vy / vx).atan() + std::f64::consts::PI / 2.;
        let phi: f64 = 2. * vz.acos();

        let ix = utils::get_index(point.position[0], &self.x_bounds);
        let iy = utils::get_index(point.position[1], &self.y_bounds);
        let iz = utils::get_index(point.position[2], &self.z_bounds);
        let ie = utils::get_index(point.energy, &self.energy_bounds);
        let it = utils::get_index(theta % (2. * std::f64::consts::PI), &self.theta_bounds);
        let ip = utils::get_index(phi % (2. * std::f64::consts::PI), &self.phi_bounds);

        if ix == None || iy == None || iz == None || ie == None || it == None || ip == None {
            return None;
        }

        Some([ix.unwrap(), iy.unwrap(), iz.unwrap(), iz.unwrap(), it.unwrap(), ip.unwrap()])
    }

    fn get_importance(&self, point: &Point) -> f64 {
        let index = self.get_index(&point);
        if index.is_some() {
            // if cell already visited, return importance
            if self.num_visits[index.unwrap()] > 0 {
                return self.importances[index.unwrap()];

            }
            // TODO finish that
            // if direction never visited, average locally scalar importance
            // else {

            //     let mut importance = 0.; //self.importances[index.unwrap()];
            //     for l in 0..self.num_cells[4]{
            //         for m in 0..self.num_cells[5]{
            //             let local_index = [index.unwrap()[0],
            //                                index.unwrap()[1],
            //                                index.unwrap()[2],
            //                                index.unwrap()[3], m, l];
            //             importance += self.importances[local_index];
            //         }
            //     }
            //     importance /= (self.num_cells[4] * self.num_cells[5]) as f64;
            //     return importance;
            // }
        }
        return std::f64::MIN;
    }
    fn score(&mut self, point: Arc<Mutex<trajectory::Point>>) {
        let point = point.lock().unwrap();
        let index = self.get_index(&point);
        if index.is_some() {
            self.scored_importances_num[index.unwrap()].add_value(point.cumulated_score);
            self.scored_importances_denom[index.unwrap()].add_value(point.weight);
            self.num_visits[index.unwrap()] += 1;
        }
    }
    fn is_some(&self) -> bool {
        true
    }
    fn compute_importances_from_score(&mut self) {
        for (i, j, k, l, m, n) in iproduct!(
            0..self.x_bounds.len() - 1,
            0..self.y_bounds.len() - 1,
            0..self.z_bounds.len() - 1,
            0..self.energy_bounds.len() - 1,
            0..self.theta_bounds.len() - 1,
            0..self.phi_bounds.len() - 1
        ) {
            let index = [i, j, k, l, m, n];
            if self.scored_importances_denom[index].get_mean() > 0. {
                self.importances[index] = self.scored_importances_num[index].get_mean()
                    / self.scored_importances_denom[index].get_mean();
            }
        }
    }
    fn write_to_file(&self, file_name: String) -> Result<(), Error> {

        let mut output_file = File::create(file_name)?;

        // loop on the whole array of importances
        let mut num_visits = Vec::new();
        let mut num = Vec::new();
        let mut denom = Vec::new();
        let mut importance = Vec::new();
        for (i, j, k, l, m, n) in iproduct!(
            0..self.x_bounds.len() - 1,
            0..self.y_bounds.len() - 1,
            0..self.z_bounds.len() - 1,
            0..self.energy_bounds.len() - 1,
            0..self.theta_bounds.len() - 1,
            0..self.phi_bounds.len() - 1
        ) {
            let index = [i, j, k, l, m, n];
            num_visits.push(self.num_visits[index]);
            num.push(self.scored_importances_num[index].get_mean());
            denom.push(self.scored_importances_denom[index].get_mean());
            importance.push(
                self.scored_importances_num[index].get_mean()
                    / self.scored_importances_denom[index].get_mean(),
            );
        }

        let importance_map_json = json!({
            "type": "map",
            "x_bounds": self.x_bounds,
            "y_bounds": self.y_bounds,
            "z_bounds": self.z_bounds,
            "energy_bounds": self.energy_bounds,
            "theta_bounds": self.theta_bounds,
            "phi_bounds": self.phi_bounds,
            "num_visits": num_visits,
            "num": num,
            "denom": denom,
            "importance": importance
        });
        let _response_string = format!(
            "{}",
            serde_json::to_string_pretty(&importance_map_json).unwrap()
        );
        write!(output_file, "{}", _response_string).unwrap();

        Ok(())
    }

    fn build(&mut self) {}
    fn train(&mut self) {}

    fn collect_normalisation(&mut self, norm: f64) {
        for tally in self.scored_importances_num.iter_mut() {
            tally.current_value /= norm;
        }
        for tally in self.scored_importances_denom.iter_mut() {
            tally.current_value /= norm;
        }
    }

    fn prepare_next_batch(&mut self) {
        for tally in self.scored_importances_num.iter_mut() {
            tally.prepare_next_batch();
        }
        for tally in self.scored_importances_denom.iter_mut() {
            tally.prepare_next_batch();
        }
    }

    fn read_from_file(file_name: String) -> Self {
        let file = File::open(file_name).expect("file should open read only");
        let idict: serde_json::Value =
            serde_json::from_reader(file).expect("file should be proper JSON");
        let x_bounds: Vec<f64> = idict["x_bounds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_f64().unwrap())
            .collect();
        let y_bounds: Vec<f64> = idict["y_bounds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_f64().unwrap())
            .collect();
        let z_bounds: Vec<f64> = idict["z_bounds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_f64().unwrap())
            .collect();
        let energy_bounds: Vec<f64> = idict["energy_bounds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_f64().unwrap())
            .collect();
        let theta_bounds: Vec<f64> = idict["theta_bounds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_f64().unwrap())
            .collect();
        let phi_bounds: Vec<f64> = idict["phi_bounds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_f64().unwrap())
            .collect();

        let num_visits: Vec<usize> = idict["num_visits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_u64().unwrap() as usize)
            .collect();
        let num: Vec<f64> = idict["num"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_f64().unwrap())
            .collect();
        let denom: Vec<f64> = idict["denom"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_f64().unwrap())
            .collect();
        let read_importances: Vec<f64> = idict["importance"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_f64().unwrap())
            .collect();

        let num_cells = [
            x_bounds.len() - 1,
            y_bounds.len() - 1,
            z_bounds.len() - 1,
            energy_bounds.len() - 1,
            theta_bounds.len() - 1,
            phi_bounds.len() - 1,
        ];
        let mut visits = Array6::<usize>::zeros((
            num_cells[0],
            num_cells[1],
            num_cells[2],
            num_cells[3],
            num_cells[4],
            num_cells[5],
        ));
        let mut scored_importances_num = Array6::<Tally>::zeros((
            num_cells[0],
            num_cells[1],
            num_cells[2],
            num_cells[3],
            num_cells[4],
            num_cells[5],
        ));
        let mut scored_importances_denom = Array6::<Tally>::zeros((
            num_cells[0],
            num_cells[1],
            num_cells[2],
            num_cells[3],
            num_cells[4],
            num_cells[5],
        ));
        let mut importances = Array6::<f64>::zeros((
            num_cells[0],
            num_cells[1],
            num_cells[2],
            num_cells[3],
            num_cells[4],
            num_cells[5],
        ));

        let mut id = 0;
        for (i, j, k, l, m, n) in iproduct!(
            0..num_cells[0],
            0..num_cells[1],
            0..num_cells[2],
            0..num_cells[3],
            0..num_cells[4],
            0..num_cells[5]
        ) {
            let index = [i, j, k, l, m, n];
            visits[index] = num_visits[id];
            scored_importances_num[index].mean = num[id];
            scored_importances_denom[index].mean = denom[id];
            importances[index] = read_importances[id];
            id += 1;
        }

        Self {
            x_bounds: x_bounds,
            y_bounds: y_bounds,
            z_bounds: z_bounds,
            energy_bounds: energy_bounds,
            theta_bounds: theta_bounds,
            phi_bounds: phi_bounds,
            scored_importances_num: scored_importances_num,
            scored_importances_denom: scored_importances_denom,
            importances: importances,
            num_visits: visits,
        }
    }
}
impl DirectionalImportanceMap {
    pub fn new(
        x_bounds: Vec<f64>,
        y_bounds: Vec<f64>,
        z_bounds: Vec<f64>,
        energy_bounds: Vec<f64>,
        theta_bounds: Vec<f64>,
        phi_bounds: Vec<f64>,
    ) -> Self {
        // read new importance map from input json file
        let num_cells = [
            x_bounds.len() - 1,
            y_bounds.len() - 1,
            z_bounds.len() - 1,
            energy_bounds.len() - 1,
            theta_bounds.len() - 1,
            phi_bounds.len() - 1,
        ];

        let importances = Array6::<f64>::zeros((
            num_cells[0],
            num_cells[1],
            num_cells[2],
            num_cells[3],
            num_cells[4],
            num_cells[5],
        ));

        let num = Array6::<Tally>::zeros((
            num_cells[0],
            num_cells[1],
            num_cells[2],
            num_cells[3],
            num_cells[4],
            num_cells[5],
        ));

        let denom = Array6::<Tally>::zeros((
            num_cells[0],
            num_cells[1],
            num_cells[2],
            num_cells[3],
            num_cells[4],
            num_cells[5],
        ));

        let nvisits = Array6::<usize>::zeros((
            num_cells[0],
            num_cells[1],
            num_cells[2],
            num_cells[3],
            num_cells[4],
            num_cells[5],
        ));

        Self {
            importances: importances,
            x_bounds: x_bounds,
            y_bounds: y_bounds,
            z_bounds: z_bounds,
            energy_bounds: energy_bounds,
            theta_bounds: theta_bounds,
            phi_bounds: phi_bounds,
            scored_importances_num: num,
            scored_importances_denom: denom,
            num_visits: nvisits,
        }
    }
}
