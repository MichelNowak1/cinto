
use std::io::Error;

use crate::common::importance_map::importance::Importance;
use crate::monte_carlo::trajectory::Point;

use std::sync::{Arc, Mutex};

pub struct Null {}

impl Importance for Null {
    fn get_importance(&self, point: &Point) -> f64 {
        0.
    }
    fn is_some(&self) -> bool {
        false
    }
    fn score(&mut self, _point: Arc<Mutex<Point>>) {}
    fn get_index(&self, _point: &Point) -> Option<[usize; 6]> {
        let _index: [usize; 6] = [0, 0, 0, 0, 0, 0];
        Some(_index)
    }
    fn compute_importances_from_score(&mut self) {}
    fn read_from_file(file_name: String) -> Self {
        Self{}
    }
    fn write_to_file(&self, file_name: String) -> Result<(), Error> {
        Ok(())
    }
    fn build(&mut self) {}
    fn train(&mut self) {}
    fn prepare_next_batch(&mut self) {}
    fn collect_normalisation(&mut self, _source_norm: f64) {}
}
