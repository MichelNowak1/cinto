

use std::io::Error;

use crate::common::importance_map::importance::Importance;
use crate::monte_carlo::trajectory::Point;

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

#[pyclass]
pub struct FunctionalImportance {
    // function importance structure
    pub origin: Vec<f64>,
    pub itype: String,
}

#[pymethods]
impl FunctionalImportance{
    #[new]
    pub fn new(origin: Vec<f64>, itype: String) -> Self{
        Self {
            origin: origin,
            itype: itype
        }
    }
}
impl Importance for FunctionalImportance {
    fn get_importance(&self, point: &Point) -> f64 {
        // get particle-origin distance
        let mut importance: f64 = (
              (point.position[0] - self.origin[0]).powi(2)
            + (point.position[1] - self.origin[1]).powi(2)
            + (point.position[2] - self.origin[2]).powi(2))
        .sqrt();
        if self.itype == "attractor" {
            importance = 1. / importance;
        }
        importance
    }
    fn is_some(&self) -> bool {
        true
    }
    fn score(&mut self, _point: Arc<Mutex<Point>>) {}
    fn get_index(&self, _point: &Point) -> Option<[usize; 6]> {
        let _index: [usize; 6] = [0, 0, 0, 0, 0, 0];
        Some(_index)
    }
    fn compute_importances_from_score(&mut self) {}
    fn read_from_file(file_name: String) -> Self{
        Self{
            origin: vec![0., 0., 0.],
            itype: "attractor".to_string(),
        }
    }
    fn write_to_file(&self, file_name: String) -> Result<(), Error> {
        Ok(())
    }
    fn build(&mut self) {}
    fn train(&mut self) {}
    fn prepare_next_batch(&mut self) {}
    fn collect_normalisation(&mut self, _source_norm: f64) {}
}
