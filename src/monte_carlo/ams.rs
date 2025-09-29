use crate::common::cross_section_library::Interaction;
use crate::monte_carlo::particle;
use crate::monte_carlo::trajectory;

use pyo3::prelude::*;
use rayon::prelude::*;
use rand::prelude::*;

use rand::thread_rng;

/// Adaptive Multilevel Splitting
///
/// # Attributes
/// * `current_iteration` current iteration of the algorithm
/// * `level` current AMS level of the algorithm
/// * `global_weight` current weight assigned to particles active in current iteration
/// * `batch_size` size of the batch
/// * `notes` vector of notes (importances) assigned to each active trajectory
/// * `num_splits` AMS splitting parameter gives the number of particles to be splitted per
///                iteration
/// * `replicas` vector containing the freshly generated splitted particles at the end of current
///              iteration
/// * `trajectories` vector containing the trajectories of the AMS algorithm
/// * `current_trajectory_index` index tracking the current trajectory index, used for
///                              backpropagation
#[pyclass]
pub struct AMS {
    #[pyo3(get,set)]
    pub current_iteration: i32,
    #[pyo3(get,set)]
    pub level: f64,
    #[pyo3(get,set)]
    pub global_weight: f64,
    #[pyo3(get,set)]
    pub k_split: usize,
    #[pyo3(get,set)]
    pub batch_size: usize,
    #[pyo3(get,set)]
    pub notes: Vec<f64>,
    #[pyo3(get,set)]
    pub num_splits: usize,
    #[pyo3(get,set)]
    pub replicas: Vec<particle::Particle>,
    #[pyo3(get,set)]
    pub trajectories: Vec<trajectory::Trajectory>,
    #[pyo3(get,set)]
    pub current_trajectory_index: i64,
}

#[pymethods]
impl AMS {
    #[new]
    pub fn new(
        batch_size:  usize,
        num_particles_per_batch: usize,
        k_split: usize)-> Self{
        Self{
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
        }
    }
    /// Iterates onces over the AMS algorithm
    ///
    /// # Arguments
    /// * `trajectories` set of trajectories used for AMS iteration
    ///
    /// # Returns
    /// * `continue_iterating` a boolean: if false, AMS has reached its convergence criterion
    pub fn iterate(
        &mut self,
        trajectories_input: Vec<trajectory::Trajectory>
        ) -> bool {

        let verbose = false;

        let mut trajectories = trajectories_input.clone();

        for trajectory in trajectories.iter_mut() {
            trajectory.id = self.current_trajectory_index;
            self.current_trajectory_index += 1;
        }
        self.trajectories.append(&mut trajectories);

        self.replicas.clear();
        self.notes.clear();

        // first, get importance trajectories into notes
        for trajectory in self.trajectories.iter_mut() {
            if trajectory.active {
                self.notes.push(trajectory.importance);
            }
        }

        // sort notes
        self.notes.par_sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.level = self.notes[self.k_split - 1];

        let level = self.level;

        let mut num_splits = 0;
        for trajectory in self.trajectories.iter_mut() {
            if trajectory.active && (trajectory.importance <= level) {
                trajectory.ams_weight = self.global_weight;
                trajectory.active = false;
                num_splits += 1;
            }
        }
        if verbose { 
            println!(
                "ams {} {} {} {}",
                self.current_iteration,
                num_splits,
                self.batch_size,
                self.level);
        }

        if num_splits == self.batch_size {
            return false;
        }

        self.global_weight *= 1. - (num_splits as f64) / (self.batch_size as f64);

        for _ in 0..num_splits {
            self.generate_new_replicas();
        }

        self.current_iteration += 1;

        true
    }

    /// Generates new replicas from current trajectories
    pub fn generate_new_replicas(&mut self) {
        // Chose trajectory on which to split
        let mut rng = thread_rng();

        let mut trajectory: &trajectory::Trajectory;
        loop {
            trajectory = self.trajectories.choose(&mut rng).unwrap();
            if trajectory.active {
                break;
            }
        }

        // get branch
        let branch = &trajectory.branches[0].lock().unwrap();
        let branch_id = 0;

        // get point on which to split
        let mut point_id = 0;

        loop {
            let point = branch.points[point_id].lock().unwrap();
            let importance = point.importance;
            let last_interaction = point.last_interaction;
            drop(point);

            if (last_interaction == Interaction::ElasticScattering
                || last_interaction == Interaction::Source
                || last_interaction == Interaction::Reflection
                || last_interaction == Interaction::Split)
                && importance > self.level 
            {
                break;
            }
            if point_id == branch.points.len()-1{
                break
            }
            point_id += 1;
        }

        let p = branch.points[point_id].lock().unwrap();

        // Create particle that will start from this point
        self.replicas.push(particle::Particle {
            position: p.position,
            direction: p.direction,
            energy: p.energy,
            time: p.time,
            weight: p.weight,
            importance: p.importance,
            contribution: 0.,
            is_colliding: p.is_collision,
            is_absorbed: p.is_absorbed,
            is_crossing: p.is_crossing,
            mother_trajectory_id: trajectory.id as i32,
            mother_branch_id: branch_id as i32,
            rank_in_mother_branch: point_id as i32,
            last_interaction: Interaction::Split,
            initial_energy: p.energy,
        });
    }
    /// Prepares next batch by resetting key AMS parameters
    pub fn prepare_next_batch(&mut self) {
        self.global_weight = 1.;
        self.current_iteration = 0;
        self.replicas.clear();
        self.notes.clear();
        self.level = 0.;
        self.trajectories.clear();
        self.current_trajectory_index = 0;
    }
}
