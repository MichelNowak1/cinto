
use itertools::iproduct;
use ndarray::prelude::*;
use serde_json::json;
use std::fs::File;
use std::io::Error;
use std::io::Write;

use crate::common::importance_map::importance::Importance;
use crate::monte_carlo::trajectory::Point;
use crate::monte_carlo::tally::Tally;
use crate::monte_carlo::trajectory;
use crate::common::utils;

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
#[pyclass]
pub struct ScalarImportanceMap {
    // importance map structure
    pub x_bounds: Vec<f64>,
    pub y_bounds: Vec<f64>,
    pub z_bounds: Vec<f64>,
    pub energy_bounds: Vec<f64>,

    // importances
    pub importances: Array4<f64>,
    pub scored_importances_num: Array4<Tally>,
    pub scored_importances_denom: Array4<Tally>,
    pub num_visits: Array4<usize>,
}

impl Importance for ScalarImportanceMap {
    /*
     * @method get_6d_index
     * @args trajectory::Point
     * @returns index at which the point is
     */
    fn get_index(&self, point: &trajectory::Point) -> Option<[usize; 6]> {
        let ix = utils::get_index(point.position[0], &self.x_bounds);
        let iy = utils::get_index(point.position[1], &self.y_bounds);
        let iz = utils::get_index(point.position[2], &self.z_bounds);
        let ie = utils::get_index(point.energy, &self.energy_bounds);
        if ix == None || iy == None || iz == None || ie == None {
            return None;
        }

        Some([ix.unwrap(), iy.unwrap(), iz.unwrap(), ie.unwrap(), 0, 0])
    }

    fn get_importance(&self, point: &Point) -> f64 {
        let index6 = self.get_index(&point);
        //  println!("{:?}", index6);

        if index6.is_some() {
            let index = [
                index6.unwrap()[0],
                index6.unwrap()[1],
                index6.unwrap()[2],
                index6.unwrap()[3],
            ];

            return self.importances[index];
        }
        return std::f64::MIN;
    }
    fn is_some(&self) -> bool {
        true
    }
    /*
     * @method score
     * @brief scores a point on the map
     * @args trajectory::Point
     * @returns index at which the point is
     */
    fn score(&mut self, point: Arc<Mutex<trajectory::Point>>) {
        // let p = point.lock().unwrap();
        // let r:f64 = (p.position[0].powf(2.) + p.position[1].powf(2.) + p.position[2].powf(2.)).sqrt();
        // println!("NUM {} {}", r, p.cumulated_score);
        // println!("DENOM {} {}", r, p.weight);
        // drop(p);

        let index6 = self.get_index(&point.lock().unwrap());
        if index6.is_some() {
            let index = [
                index6.unwrap()[0],
                index6.unwrap()[1],
                index6.unwrap()[2],
                index6.unwrap()[3],
            ];
                self.scored_importances_num[index].add_value(point.lock().unwrap().cumulated_score);
                self.scored_importances_denom[index].add_value(point.lock().unwrap().weight);
                self.num_visits[index] += 1;
        }
    }
    fn compute_importances_from_score(&mut self) {
        for (i, j, k, l) in iproduct!(
            0..self.x_bounds.len() - 1,
            0..self.y_bounds.len() - 1,
            0..self.z_bounds.len() - 1,
            0..self.energy_bounds.len() -1
        ) {
            let index = [i, j, k, l];
            if self.scored_importances_denom[index].get_mean() > 0. {
                self.importances[index] = self.scored_importances_num[index].get_mean()
                    / self.scored_importances_denom[index].get_mean();
            }
        }
    }
    fn write_to_file(&self, file_name: String) -> Result<(), Error> {

        let mut output_file = File::create(file_name)?;

        // loop on the whole array of importances
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];
        let z: Vec<f64> = vec![];
        let e: Vec<f64> = vec![];
        let mut num_visits: Vec<usize> = vec![];
        let mut num: Vec<f64> = vec![];
        let mut denom: Vec<f64> = vec![];
        let mut importance: Vec<f64> = vec![];
        for (i, j, k, l) in iproduct!(
            0..self.x_bounds.len() - 1,
            0..self.y_bounds.len() - 1,
            0..self.z_bounds.len() - 1,
            0..self.energy_bounds.len() -1
        ) {
            let index = [i, j, k, l];

            num_visits.push(self.num_visits[index]);
            num.push(self.scored_importances_num[index].get_mean());
            denom.push(self.scored_importances_denom[index].get_mean());

            if self.scored_importances_denom[index].get_mean() > 0. {
                importance.push(
                    self.scored_importances_num[index].get_mean()
                    / self.scored_importances_denom[index].get_mean(),
                );
            } else {
                importance.push(0.);
            }
        }

        let importance_map_json = json!({
            "type": "scalar",
            "x_bounds": self.x_bounds,
            "y_bounds": self.y_bounds,
            "z_bounds": self.z_bounds,
            "energy_bounds": self.energy_bounds,
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
    fn read_from_file(file_name: String) -> Self{
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
        // let num_visits: Vec<usize> = idict["num_visits"]
        //     .as_array()
        //     .unwrap()
        //     .iter()
        //     .map(|f| f.as_u64().unwrap() as usize)
        //     .collect();
        // let num: Vec<f64> = idict["num"]
        //     .as_array()
        //     .unwrap()
        //     .iter()
        //     .map(|f| f.as_f64().unwrap())
        //     .collect();
        // let denom: Vec<f64> = idict["denom"]
        //     .as_array()
        //     .unwrap()
        //     .iter()
        //     .map(|f| f.as_f64().unwrap())
        //     .collect();
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
        ];
        let visits = Array4::<usize>::zeros((
            num_cells[0],
            num_cells[1],
            num_cells[2],
            num_cells[3]
        ));
        let scored_importances_num = Array4::<Tally>::zeros((
            num_cells[0],
            num_cells[1],
            num_cells[2],
            num_cells[3],
        ));
        let scored_importances_denom = Array4::<Tally>::zeros((
            num_cells[0],
            num_cells[1],
            num_cells[2],
            num_cells[3],
        ));
        let mut importances = Array4::<f64>::zeros((
            num_cells[0],
            num_cells[1],
            num_cells[2],
            num_cells[3],
        ));

        let mut id = 0;
        for (i, j, k, l) in iproduct!(
            0..num_cells[0],
            0..num_cells[1],
            0..num_cells[2],
            0..num_cells[3]
        ) {
            let index = [i, j, k, l];
            importances[index] = read_importances[id];
            id += 1;
        }
        Self {
            x_bounds: x_bounds,
            y_bounds: y_bounds,
            z_bounds: z_bounds,
            energy_bounds: energy_bounds,
            scored_importances_num: scored_importances_num,
            scored_importances_denom: scored_importances_denom,
            importances: importances,
            num_visits: visits,
        }
    }
}
impl ScalarImportanceMap {
    pub fn build_new(
        x_bounds: Vec<f64>,
        y_bounds: Vec<f64>,
        z_bounds: Vec<f64>,
        energy_bounds: Vec<f64>
    ) -> Self {
        // read new importance map from input json file
        let num_cells = [x_bounds.len() - 1, y_bounds.len() - 1, z_bounds.len() - 1, energy_bounds.len() - 1];
        let importances = Array4::<f64>::zeros((num_cells[0], num_cells[1], num_cells[2], num_cells[3]));
        let num = Array4::<Tally>::zeros((num_cells[0], num_cells[1], num_cells[2], num_cells[3]));
        let denom = Array4::<Tally>::zeros((num_cells[0], num_cells[1], num_cells[2], num_cells[3]));
        let nvisits = Array4::<usize>::zeros((num_cells[0], num_cells[1], num_cells[2], num_cells[3]));

        return Self {
            x_bounds: x_bounds,
            y_bounds: y_bounds,
            z_bounds: z_bounds,
            energy_bounds: energy_bounds,
            importances: importances,
            scored_importances_num: num,
            scored_importances_denom: denom,
            num_visits: nvisits,
        };
    }
}
#[pymethods]
impl ScalarImportanceMap {
    #[new]
    pub fn new(file_name: String) -> Self{
        ScalarImportanceMap::read_from_file(file_name)

    }
    pub fn pyget_importance(&self, point: &Point) -> f64{
        self.get_importance(point)
    }
}
