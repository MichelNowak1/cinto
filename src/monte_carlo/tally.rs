use core::ops::Add;
use pyo3::prelude::*;

extern crate num_traits;

#[derive(Clone)]
pub enum TallyType {
    Collision,
    Path,
    Absorption,
}

/// Tally
///
/// # Attributes
/// * `tally_type` type of the tally
/// * `num_values` number of values currently stored in tally
/// * `current_value` current value of the tally
/// * `mean` current mean of the tally
/// * `standard` current mean of the tally
#[derive(Clone)]
#[pyclass]
pub struct Tally {
    pub tally_type: TallyType,
    pub num_values: usize,
    pub current_value: f64,
    pub mean: f64,
    pub standard_deviation: f64,
    pub values: Vec<f64>,
    pub sum: f64,
    pub sum_squarred: f64,
    pub len: usize,
}

impl num_traits::identities::Zero for Tally {
    fn zero() -> Tally {
        Tally {
            tally_type: TallyType::Collision,
            num_values: 0,
            current_value: 0.,
            mean: 0.,
            standard_deviation: 0.,
            sum: 0.,
            sum_squarred: 0.,
            len: 0,
            values: vec![0.],
        }
    }
    fn is_zero(&self) -> bool {
        if self.values.len() <= 1 {
            return true;
        }
        return false;
    }
}

impl Add for Tally {
    type Output = Self;
    fn add(self, _other: Self) -> Self {
        Self {
            tally_type: TallyType::Collision,
            num_values: self.num_values,
            current_value: 0.,
            mean: 0.,
            standard_deviation: 0.,
            sum: 0.,
            sum_squarred: 0.,
            len: 0,
            values: vec![0.],
        }
    }
}
#[pymethods]
impl Tally {
    #[new]
    pub fn new() -> Self {
        Self {
            tally_type: TallyType::Collision,
            num_values: 0,
            current_value: 0.,
            mean: 0.,
            standard_deviation: 0.,
            sum: 0.,
            sum_squarred: 0.,
            len: 0,
            values: vec![0.],
        }
    }
}

impl Tally {

    pub fn add_value(&mut self, value: f64) {
        self.current_value += value;
    }

    pub fn get_current_value(&self) -> f64 {
        self.current_value
    }

    pub fn get_mean(&self) -> f64 {
        if self.len > 0 {
            self.sum / (self.len as f64)
        } else {
            return self.current_value;
        }
    }

    /// in percentage
    pub fn get_standard_deviation(&self) -> f64 {
        let mean = self.get_mean();
        let len = self.len as f64;

        if len < 1. {
            return 100.;
        }

        let variance = (self.sum_squarred - mean * mean) / (len-1.) / len;

        let standard_deviation_of_mean = variance.sqrt();

        100.*standard_deviation_of_mean/mean
    }

    pub fn reset(&mut self) {
        self.num_values = 0;
        self.current_value = 0.;
        self.mean = 0.;
        self.sum_squarred = 0.;
        self.standard_deviation = 0.;
        self.values.clear();
    }
    pub fn prepare_next_batch(&mut self) {
        self.sum += self.current_value;
        self.sum_squarred += self.current_value * self.current_value ;
        self.values.push(self.current_value);
        self.len += 1;
        self.current_value = 0.;
    }
}
