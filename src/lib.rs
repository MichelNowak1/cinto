use pyo3::prelude::*;

///! Cinto is a particle transport playground designed to easily experiment new algorithms.

/// Input module
pub mod input;

/// Common module:
/// contains all structures common to Monte Carlo and Deterministic calculation modes
pub mod common;

/// Monte Carlo module:
/// contains all structures to solve the problem with the Monte Carlo method
pub mod monte_carlo;

/// Deterministic module:
/// contains all structures to solve the problem with Deterministic methods
pub mod deterministic;

use common::geometry::Geometry;
use common::cross_section_library::CrossSectionLibrary;
use common::detector::Detector;
use common::source::Source;
use common::importance_map::scalar::ScalarImportanceMap;
use common::importance_map::functional::FunctionalImportance;
use monte_carlo::trajectory::Trajectory;
use monte_carlo::trajectory::Point;
use monte_carlo::particle::Particle;
use monte_carlo::solver::MonteCarloSolver;
use monte_carlo::ams::AMS;

#[pymodule]
fn _cinto(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<CrossSectionLibrary>()?;
    m.add_class::<Geometry>()?;
    m.add_class::<Detector>()?;
    m.add_class::<Particle>()?;
    m.add_class::<MonteCarloSolver>()?;
    m.add_class::<AMS>()?;
    m.add_class::<ScalarImportanceMap>()?;
    m.add_class::<FunctionalImportance>()?;
    m.add_class::<Source>()?;
    m.add_class::<Trajectory>()?;
    m.add_class::<Point>()?;
    Ok(())
}
