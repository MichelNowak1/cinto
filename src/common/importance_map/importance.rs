use std::io::Error;

use crate::monte_carlo::trajectory::Point;

use std::sync::{Arc, Mutex};

pub trait Importance {
    fn get_importance(&self, point: &Point) -> f64;
    fn is_some(&self) -> bool;
    fn score(&mut self, point: Arc<Mutex<Point>>);
    fn get_index(&self, point: &Point) -> Option<[usize; 6]>;
    fn compute_importances_from_score(&mut self);
    fn read_from_file(file_name: String) -> Self where Self: Sized;
    fn write_to_file(&self, file_name: String) -> Result<(), Error>;
    fn build(&mut self);
    fn train(&mut self);
    fn collect_normalisation(&mut self, source_norm: f64);
    fn prepare_next_batch(&mut self);
}
