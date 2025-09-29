use crate::common::calculation_type::CalculationType;
use crate::common::cross_section_library::CrossSectionLibrary;
use crate::common::detector;
use crate::common::geometry;
use crate::common::importance_map::importance::Importance;
use crate::common::mesh;
use crate::common::profiler;
use crate::common::source;

use crate::monte_carlo::ams;
use crate::monte_carlo::homogenizer::Homogenizer;
use crate::monte_carlo::particle;
use crate::monte_carlo::strategy::Strategy;
use crate::monte_carlo::trajectory;
use crate::monte_carlo::trajectory::Trajectory;
use crate::monte_carlo::transport_analog;
use crate::monte_carlo::transport_mode::TransportMode;
use crate::monte_carlo::variance_reduction::VarianceReductionMethod;

use colored::*;
use console::style;
use rand::Rng;
use serde_json::json;
use std::cell::RefCell;
use std::fs::File;
use std::io::Write;
use std::rc::Rc;

use pyo3::prelude::*;
use rayon::prelude::*;

/// MonteCarloSolver
///
/// # Attributes
/// * `num_batches` number of batches to be simulated
/// * `num_particles_per_batch` number of particles to be simulated in each batch
///
///
/// * `ams` Adaptive Multilevel Splitting manager for shielding calculations
/// * `importance_sampling` if true, importance sampling will be activated
/// * `scoring_importance` if true, provided scored_importance_map will be updated

#[pyclass]
pub struct MonteCarloSolver {
    // general
    pub calculation_type: CalculationType,
    pub num_batches: usize,
    pub num_particles_per_batch: usize,

    // variance reduction
    pub ams: Option<ams::AMS>,
    pub importance_sampling: bool,
    pub scoring_importance: bool,
    pub transport_mode: TransportMode,
}

#[pymethods]
impl MonteCarloSolver {
    /// Creates a new Monte Carlo Solver
    ///
    /// # Arguments
    /// * `input` json_input to be parsed to retrieve Monte Carlo solver parameters
    ///
    /// # Returns
    /// * `MonteCarloSolver` a MonteCarloSolver instance
    #[new]
    pub fn new(
        calculation_type: String,
        num_batches: usize,
        num_particles_per_batch: usize,
        variance_reduction_method: String,
        transport_mode: String,
    ) -> MonteCarloSolver {

        let ct = CalculationType::Shielding;

        let tm = if transport_mode.eq("importance_sampling"){
            TransportMode::ImportanceSampling
        } else {
            TransportMode::Analog
        };

        let vrm = if variance_reduction_method.eq("ams") {
            VarianceReductionMethod::AMS
        } else if variance_reduction_method.eq("importance_sampling") {
            VarianceReductionMethod::ImportanceSampling
        } else if variance_reduction_method.eq("ams_importance_sampling") {
            VarianceReductionMethod::AMSImportanceSampling
        } else {
            VarianceReductionMethod::Analog
        };


        // Get variance reduction method
        let mut ams = None;
        let mut importance_sampling = false;

        if vrm == VarianceReductionMethod::AMS
            || vrm == VarianceReductionMethod::AMSImportanceSampling
        {
            println!(
                "{} {} {}",
                style("->").bold().dim(),
                "\u{26F3}",
                "Initializing Adaptive Multilevel Splitting..."
                    .bold()
                    .blue()
            );
            let k_split = 100;

            // create AMS 
            ams = Some(ams::AMS {
                current_iteration: 0,
                level: -1.,
                global_weight: 1.,
                k_split: k_split,
                notes: Vec::new(),
                batch_size: num_particles_per_batch,
                num_splits: 0,
                replicas: Vec::new(),
                trajectories: Vec::new(),
                current_trajectory_index: 0,
            });
        }
        if vrm == VarianceReductionMethod::ImportanceSampling
            || vrm == VarianceReductionMethod::AMSImportanceSampling
        {
            importance_sampling = true;
        }

        MonteCarloSolver {
            calculation_type: ct,
            num_batches: num_batches,
            num_particles_per_batch: num_particles_per_batch,
            ams: ams,
            importance_sampling: importance_sampling,
            scoring_importance: false,
            transport_mode: tm,
        }
    }
}

impl MonteCarloSolver {
    /// Solve the problem
    ///
    /// # Arguments
    /// * `output_file` output file in which detectors responses need to be written
    /// * `geometry` geometry of the problem
    /// * `cross_section_library` cross section library to be used for isotopes
    /// * `sources` list of sources of particles
    /// * `detectors` list of detectors where to score a response
    /// * `imortance_map` importance map use for variance reduction techniques
    pub fn solve(
        &mut self,
        output_file: &mut File,
        geometry: &geometry::Geometry,
        cross_section_library: &CrossSectionLibrary,
        sources: &Vec<source::Source>,
        detectors: &mut Vec<detector::Detector>,
        meshes: &mut Vec<Box<dyn mesh::Mesh>>,
        importance_map: Rc<RefCell<dyn Importance>>,
        scored_importance_map: Rc<RefCell<dyn Importance>>,
        strategy: &mut Option<Strategy>,
        homogenizer: &mut Box<dyn Homogenizer>,
    ) {
        println!(
            "{} {} {}",
            style("->").bold().dim(),
            "\u{269B}",
            " Running Monte Carlo solver...".bold().green()
        );

        // init main batch progress bar
        let mut progress_bar = profiler::CintoProgressBar::new(self.num_batches);

        // start profiler
        let solver_profiler = profiler::Profiler::new();

        // activate importance map scoring is scored_importance_map is provided
        if scored_importance_map.borrow().is_some() {
            self.scoring_importance = true
        }

        // Loop on batches
        for batch_index in 0..self.num_batches {
            // let batch_profiler = profiler::Profiler::new();
            progress_bar.update(batch_index as u64);

            // initialize sources depending on use case type
            let particles = 
            {
                let mut particles = vec![];
                for source in sources{
                    particles.append(&mut self.sample_sources(source));
                }
                particles
            };

            // transport initial source particles and secondary ones
            let (produced_particles, mut trajectories) =
                self.transport_particles(
                    particles,
                    geometry,
                    &cross_section_library,
            );

            // iterate on ams trajectories
            if self.ams.is_some() {
                trajectories = self
                    .ams_iterate(
                        &mut trajectories,
                        geometry,
                        cross_section_library,
                        detectors,
                        meshes,
                        importance_map.clone()
                    )
                    .to_vec();
            }
            for trajectory in trajectories.iter() {
                for detector in detectors.iter_mut() {
                    detector.score(
                        &trajectory,
                        geometry,
                        cross_section_library);
                }
                for mesh in meshes.iter_mut() {
                    mesh.add_tally(
                        &trajectory,
                        geometry,
                        cross_section_library);
                }
            }

            // score importance with collected trajectories
            if self.scoring_importance {
                self.score_importance(
                    scored_importance_map.clone(),
                    &trajectories);
            }

            if homogenizer.is_some() {
                homogenizer.normalize(self.num_particles_per_batch as f64);
                homogenizer.update_cross_section_library();
                homogenizer.write_cross_section_library_to_file();
            }

            // collect detectors normalisation
            for detector in detectors.iter_mut() {
                detector.normalize(self.num_particles_per_batch as f64);
            }

            for mesh in meshes.iter_mut() {
                mesh.normalize(self.num_particles_per_batch as f64);
            }

            // print detectors results
            self.print_detectors(
                detectors,
                output_file,
                &solver_profiler,
                batch_index,
                strategy,
            );

            // print mesh results
            self.print_meshes(meshes);

            // dump importance map into file
            if scored_importance_map.borrow().is_some() {
                scored_importance_map
                    .borrow_mut()
                    .write_to_file("scored_importances.json".to_string())
                    .unwrap();
            }

            // prepare next batch
            self.prepare_next_batch(
                detectors,
                meshes,
                importance_map.clone(),
                scored_importance_map.clone(),
                homogenizer,
                batch_index,
                strategy,
            );
        }
        // solver_profiler.stop();
        progress_bar.finish();
    }

    /// Transports particle on one node
    ///
    /// # Arguments
    /// * `particles` vector of particles to be transported
    /// * `geometry` geometry of the problem
    /// * `cross_section_library` cross section library to be used for isotopes
    ///
    /// # Returns
    /// * `produced_particles` vector of produced particles during transport
    /// * `trajectories` vector of trajectories generated during transport
    pub fn transport_particles(
        &mut self,
        mut particles: Vec<particle::Particle>,
        geometry: &geometry::Geometry,
        cross_section_library: &CrossSectionLibrary,
    ) -> (Vec<particle::Particle>, Vec<trajectory::Trajectory>) {
        // prepare produced particles
        let mut produced_particles_ = Vec::<particle::Particle>::new();

        // prepare output trajectories
        let mut trajectories = Vec::<trajectory::Trajectory>::new();

        // iterate on batch source particles on different threads
        particles.iter_mut().for_each(|mut particle| {

            // initialise new thread for geometry
            geometry.init_thread();

            // transport one particle
            let (mut produced_particles, trajectory) =
                match self.transport_mode {
                    TransportMode::Analog => {
                        let mut trajectory = Trajectory::new(&particle);
                        let mut produced_particles_from_trajectory =
                            Vec::<particle::Particle>::new();
                        loop {
                            trajectory.add_point(&particle);
                            let done = transport_analog::step_flight(
                                &mut particle,
                                &geometry,
                                &cross_section_library
                                );
                            trajectory.add_point(&particle);

                            if done{ break; }

                            if !particle.is_colliding { continue; }

                            let (done, mut produced_particles_from_collision) =
                                transport_analog::step_collision(
                                    &mut particle,
                                    &geometry,
                                    &cross_section_library
                                    );

                            trajectory.add_point(&particle);

                            if done{ break; }

                            produced_particles_from_trajectory
                                .append(&mut produced_particles_from_collision);
                        }
                        (produced_particles_from_trajectory, trajectory)
                    }
                    TransportMode::ImportanceSampling => {
                        panic!("not implemented yet");
                    }
                };

            // retrieve produced particles
            produced_particles_.append(&mut produced_particles);

            // retrieve trajectory
            trajectories.push(trajectory);
        });
        (produced_particles_, trajectories)
    }

    /// Samples sources from list of sources
    ///
    /// # Arguments
    /// * `sources` vector of sources from which particles need to be sampled
    ///
    /// # Returns
    /// * `particles` sampled particles
    pub fn sample_sources(&mut self, sources: &source::Source) -> Vec<particle::Particle> {
        // initialize sources
        let mut particles = Vec::<particle::Particle>::new();
        for _ in 0..self.num_particles_per_batch {
            particles.push(sources.sample());
        }
        particles
    }

    /// Iterates over the Adaptive Multilevel Splitting algorithm
    ///
    /// # Arguments
    /// * `trajectories` trajectories generated from previous iteration of the algorithm
    /// * `geometry` geometry of the problem
    /// * `cross_section_library` cross section library to be used for isotopes
    /// * `detectors` list of detectors where to score a response
    /// * `meshes` list of meshes where to score a response
    /// * `imortance_map` importance map use for variance reduction techniques
    /// * `homogenizer` homogenizes cross sections into multi group formalism
    ///
    /// # Returns
    /// * `trajectories` fresh trajectories sampled during the AMS iteration
    pub fn ams_iterate(
        &mut self,
        trajectories: &mut Vec<trajectory::Trajectory>,
        geometry: &geometry::Geometry,
        cross_section_library: &CrossSectionLibrary,
        detectors: &mut Vec<detector::Detector>,
        meshes: &mut Vec<Box<dyn mesh::Mesh>>,
        importance_map: Rc<RefCell<dyn Importance>>,
    ) -> &Vec<trajectory::Trajectory> {
        // Main iteration loop of AMS
        let mut iterate_on_ams = true;

        while iterate_on_ams {
            if importance_map.borrow().is_some() {
                for trajectory in trajectories.iter_mut() {
                    let mut branch_importance = std::f64::MIN;
                    for branch in trajectory.branches.iter() {
                        for point in branch.lock().unwrap().points.iter_mut(){
                            let mut p = point.lock().unwrap();
                            if p.importance < std::f64::MAX {
                                p.importance = importance_map.borrow().get_importance(&p);
                            }
                            if p.is_absorbed{
                                p.importance = std::f64::MIN;
                            }
                            branch_importance = branch_importance.max(p.importance);
                        }
                        branch.lock().unwrap().importance = branch_importance;
                    }
                    trajectory.importance = branch_importance
                }
            }
            // perform one iteration of AMS
            iterate_on_ams = self.ams.as_mut().unwrap().iterate(trajectories.to_vec());

            // transport replicas
            let replicas = self.ams.as_mut().unwrap().replicas.clone();
            (_, *trajectories) = self.transport_particles(
                replicas,
                geometry,
                cross_section_library,
            );
            let ams_weight = self.ams.as_ref().unwrap().global_weight;
            let ams_level = self.ams.as_ref().unwrap().level;

            // collect ams weight in detectors
            for detector in detectors.iter_mut() {
                if detector.importance <= ams_level {
                    detector.normalize_ams_weight(ams_weight);
                    detector.scoring = false;
                }
            }
        }

        // Collect ams weight in meshes
        for mesh in meshes.iter_mut() {
            mesh.normalize_ams_weight(self.ams.as_ref().unwrap().global_weight);
        }
        &self.ams.as_ref().unwrap().trajectories
    }

    /// Iterates over the Adaptive Multilevel Splitting algorithm
    ///
    /// # Arguments
    /// * `scored_importance_map` importance map to be scored while transporting particles
    /// * `trajectories` trajectories from which to update the importance map scores
    pub fn score_importance(
        &self,
        scored_importance_map: Rc<RefCell<dyn Importance>>,
        trajectories: &Vec<trajectory::Trajectory>,
    ) {
        // propagate score from trajectories to importance map
        trajectory::backward_ams(
            trajectories,
            scored_importance_map.clone());

        scored_importance_map
            .borrow_mut()
            .compute_importances_from_score();
    }

    /// Prepares next batch for key modules
    ///
    /// # Arguments
    /// * `detectors` list of detectors where to score a response
    /// * `meshes` list of meshes where to score a response
    /// * `imortance_map` importance map use for variance reduction techniques
    /// * `homogenizer` homogenizes cross sections into multi group formalism
    /// * `batch_index` index of the current batch (for strategy management)
    /// * `strategy` strategy adopted for the Monte Carlo solver
    pub fn prepare_next_batch(
        &mut self,
        detectors: &mut Vec<detector::Detector>,
        meshes: &mut Vec<Box<dyn mesh::Mesh>>,
        mut importance_map: Rc<RefCell<dyn Importance>>,
        mut scored_importance_map: Rc<RefCell<dyn Importance>>,
        homogenizer: &mut Box<dyn Homogenizer>,
        batch_index: usize,
        strategy: &mut Option<Strategy>,
    ) {
        if strategy.is_some() {
            strategy.as_mut().unwrap().end_batch(
                &mut importance_map,
                &mut scored_importance_map,
                &mut self.scoring_importance,
                batch_index as u32,
            );
        }

        // prepare detectors
        for detector in detectors.iter_mut() {
            detector.prepare_next_batch();
        }

        // prepare meshes
        for mesh in meshes.iter_mut() {
            mesh.prepare_next_batch();
        }

        if homogenizer.is_some() {
            homogenizer.prepare_next_batch();
        }

        // prepare next ams batch
        if self.ams.is_some() {
            self.ams.as_mut().unwrap().prepare_next_batch();
        }
        scored_importance_map.borrow_mut().prepare_next_batch();
    }

    /// Prints detectors into output file
    ///
    /// # Arguments
    /// * `detectors` list of detectors where to score a response
    /// * `output_file` file to which the detectors' scores need to be printed
    /// * `batch_index` index of the current batch (for strategy management)
    /// * `strategy` strategy adopted for the Monte Carlo solver
    pub fn print_detectors(
        &mut self,
        detectors: &mut Vec<detector::Detector>,
        output_file: &mut File,
        solver_profiler: &profiler::Profiler,
        batch_index: usize,
        strategy: &Option<Strategy>,
    ) {
        for detector in detectors.iter() {
            if strategy.is_some() {
                let _detector_response = json!({
                    "index": batch_index.to_string(),
                    "detector": detector.volume_name.to_string(),
                    "time": solver_profiler.start.elapsed().as_millis().to_string(),
                    "current_score": format!("{:e}", detector.get_current_score()),
                    "mean": format!("{:e}", detector.get_mean()),
                    "std": format!("{:e}", detector.get_standard_deviation()),
                    "current_phase": strategy.as_ref().unwrap().get_current_phase_as_string()
                });
                let _response_string = format!(
                    "{}",
                    serde_json::to_string_pretty(&_detector_response).unwrap()
                );
                write!(output_file, "{}\n", _response_string).unwrap();
                write!(output_file, ",\n").unwrap();
            } else {
                let _detector_response = json!({
                    "index": batch_index.to_string(),
                    "detector": detector.volume_name.to_string(),
                    "time": solver_profiler.start.elapsed().as_millis().to_string(),
                    "current_score": format!("{:e}", detector.get_current_score()),
                    "mean": format!("{:e}", detector.get_mean()),
                    "std": format!("{:e}", detector.get_standard_deviation())
                });
                let _response_string = format!(
                    "{}",
                    serde_json::to_string_pretty(&_detector_response).unwrap()
                );
                write!(output_file, "{}\n", _response_string).unwrap();
                write!(output_file, ",\n").unwrap();
            }
        }
    }

    /// Prints detectors into output file
    ///
    /// # Arguments
    /// * `meshes` list of meshes where to score a response
    ///
    /// TODO: add output file
    ///
    pub fn print_meshes(&self, meshes: &Vec<Box<dyn mesh::Mesh>>) {
        for mesh in meshes.iter() {
            mesh.dump();
        }
    }
}
