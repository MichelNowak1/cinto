use crate::common::cross_section_library::Interaction;
use crate::common::importance_map::importance::Importance;
use crate::monte_carlo::particle;
use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::cell::RefCell;
extern crate rand;
use pyo3::prelude::*;

/// Point
/// multiple particles can share the same points during AMS
///
/// # Attributes
/// * `position` position of the particle
/// * `direction` direction of the particle
/// * `weight` weight of the particle
/// * `importance` importance of the point
/// * `contribution` contribution of the point to all detectors

#[derive(Debug, Clone)]
#[pyclass]
pub struct Point {
    pub position: [f64; 3],
    pub direction: [f64; 3],
    pub energy: f64,
    pub time: f64,
    pub weight: f64,
    pub importance: f64,
    pub contribution: f64,
    pub cumulated_score: f64,
    pub is_collision: bool,
    pub is_absorbed: bool,
    pub is_crossing: bool,
    pub last_interaction: Interaction,
}

#[pymethods]
impl Point {
    #[new]
    pub fn new(pos: Vec<f64>) -> Self{
        Self{
            position: [pos[0], pos[1], pos[2]],
            direction: [pos[0], pos[1], pos[2]],
            energy: 0.1,
            time: 0.,
            weight: 0.,
            importance: 0.,
            contribution: 0.,
            cumulated_score: 0.,
            is_collision: false,
            is_absorbed: false,
            is_crossing: false,
            last_interaction: Interaction::Total
        }
    }
    /*
     * @brief dump
     * dump the point attributes into a file
     */
    /// Dump information on the point
    #[allow(dead_code)]
    pub fn dump(&mut self) {
        println!(
            "point;{};{};{};{};{};{};{};{};{}",
            self.position[0],
            self.position[1],
            self.position[2],
            self.direction[0],
            self.direction[1],
            self.direction[2],
            self.energy,
            self.weight,
            self.is_collision
        );
    }
}

/// struct Branch
///
/// # Attributes
/// * points: vector of shared pointers on points
/// * branch_id: id of the branch
/// * mother_branch: id of the parent branch
/// * rank_in_mother_branch: rank at which the branch has split on the parent branch
/// * mother_track; id of the parent track
pub struct Branch {
    pub points: Vec<Arc<Mutex<Point>>>,
    id: i32,
    rank_in_mother_branch: i32,
    mother_branch: i32,
    mother_track: i32,
    pub importance: f64,
}
impl Branch {
    /// Adds a shared pointer on a point to the buffer of points
    /// also, updates the importance accordingly
    ///
    /// # Arguments
    /// * `point` pointer to point to add to branch
    pub fn add_point(&mut self, point: Arc<Mutex<Point>>) {
        let last_interaction = point.lock().unwrap().last_interaction;
        if last_interaction != Interaction::Jump && last_interaction != Interaction::Leak {
            self.importance = self.importance.max(point.lock().unwrap().importance);
        }
        self.points.push(point);
    }

    /// Retrieves a point in a branch
    ///
    /// # Arguments
    /// * The index of the point on the branch
    ///
    /// # Returns
    /// * A reference to a point in the branch
    ///
    /// # Example
    /// ```
    /// let _point_index = 0;
    /// branch.get_point(_point_index);
    /// ```
    pub fn get_point(&mut self, index: usize) -> Arc<Mutex<Point>> {
        Arc::clone(&self.points[index])
    }

    /// dumps information on the branch
    #[allow(dead_code)]
    pub fn dump(&mut self) {
        println!(
            "branch;{};{};{};{}",
            self.id, self.mother_branch, self.rank_in_mother_branch, self.mother_track,
        );
    }
}

/// Trajectory
///
/// # Attributes
/// * `branches` vector of branches of which the trajectory is buit
/// * `id` id of the trajectory
/// * `importance` importance of the trajectory
/// * `ams_weight` ams weight associated to the trajectory
#[pyclass]
pub struct Trajectory {
    pub branches: Vec<Arc<Mutex<Branch>>>,
    pub id: i64,
    pub importance: f64,
    pub ams_weight: f64,
    pub active: bool,
}

impl Clone for Trajectory {
    fn clone(&self) -> Self {
        Trajectory {
            branches: self.branches.clone(),
            id: self.id,
            importance: self.importance,
            ams_weight: self.ams_weight,
            active: self.active,
        }
    }
}

#[pymethods]
impl Trajectory {
    /// adds a branch to the trajectory
    ///
    /// # Arguments
    /// * `branch` shared pointer of branch to be added to trajectory
    #[new]
    pub fn new_point(point: Point) -> Self{
        let _branch = Branch {
            points: Vec::new(),
            id: 0,
            rank_in_mother_branch: 0,
            mother_track: 0,
            mother_branch: 0,
            importance: point.importance,
        };
        let branches = vec![Arc::new(Mutex::new(_branch))];
        Self {
            branches: branches,
            id: -1,
            importance: -1.,
            ams_weight: -1.,
            active: true,
        }
    }
}
impl Trajectory{
    pub fn new(particle: &particle::Particle) -> Self {
        let _point: Arc<Mutex<Point>> = Arc::new(Mutex::new(Point {
            position: particle.position,
            direction: particle.direction,
            energy: particle.energy,
            time: particle.time,
            weight: particle.weight,
            importance: particle.importance,
            contribution: particle.contribution,
            cumulated_score: 0.,
            is_collision: particle.is_colliding,
            is_absorbed: particle.is_absorbed,
            is_crossing: particle.is_crossing,
            last_interaction: particle.last_interaction,
        }));
        let _branch = Branch {
            points: Vec::new(),
            id: 0,
            rank_in_mother_branch: particle.rank_in_mother_branch,
            mother_track: particle.mother_trajectory_id,
            mother_branch: particle.mother_branch_id,
            importance: particle.importance,
        };
        let branches = vec![Arc::new(Mutex::new(_branch))];
        Self {
            branches: branches,
            id: -1,
            importance: -1.,
            ams_weight: -1.,
            active: true,
        }
    }

    /// adds a point to the last branch of the last trajectory
    ///
    /// # Arguments:
    /// * `particle` particle at which to store the point
    pub fn add_point(&mut self, particle: &particle::Particle) {
        let _point: Arc<Mutex<Point>> = Arc::new(Mutex::new(Point {
            position: particle.position,
            direction: particle.direction,
            energy: particle.energy,
            time: particle.time,
            weight: particle.weight,
            importance: particle.importance,
            contribution: particle.contribution,
            cumulated_score: 0.,
            is_collision: particle.is_colliding,
            is_absorbed: particle.is_absorbed,
            is_crossing: particle.is_crossing,
            last_interaction: particle.last_interaction,
        }));
        self.branches[0].lock().unwrap().add_point(_point);
    }
    pub fn get_size(&self)->usize{
        self.branches[0].lock().unwrap().points.len()
    }

    pub fn get_point(&self, point_id: usize)->Vec<f64>{
        let point = &self.branches[0].lock().unwrap().points[point_id];
        let mut coords = Vec::<f64>::new();
        for i in 0..3{
            coords.push(point.lock().unwrap().position[i]);
        }
        coords
    }
}
impl Trajectory {

    /// gets a branch on the trajectory
    ///
    /// # Arguments
    /// * `index` index of the branch on the trajectory
    ///
    /// # Returns
    /// * `branch` shared pointer to the searched branch
    pub fn get_branch(&mut self, index: usize) -> Arc<Mutex<Branch>> {
        Arc::clone(&self.branches[index])
    }
    /// gets number of branches in trajectory
    ///
    /// # Returns
    /// * `num_branches` number of branches on the trajectory
    pub fn get_nb_branches(&self) -> usize {
        self.branches.len()
    }

    /// computes the importance of the trajectory
    ///
    /// # Returns
    /// * `impotance` importance of the computed trajectory
    pub fn compute_importance(&mut self) -> f64 {
        let mut importance = std::f64::MIN;
        for branch in self.branches.iter() {
            importance = importance.max(branch.lock().unwrap().importance);
        }
        self.importance = importance;
        self.importance
    }

    /// dumps info on trajectory
    pub fn dump(&mut self) {
        println!("trajectory;{};{}", self.id, self.branches.len());
    }
}

/// Scan backwards on AMS trajectories
///
/// # Attributes
/// * `trajectories` trajectories to be scanned backwards
/// * `importance_map` importance map that needs to be scored with the adjoint response
pub fn backward_ams(
    trajectories: &Vec<Trajectory>,
    importance_map: Rc<RefCell<dyn Importance>>,
) -> Vec<Point> {
    let mut scoring_points = vec![];

    for trajectory_index in 0..trajectories.len() {
        let trajectory = &trajectories[trajectory_index];
        let mut current_trajectory_index: i32 = trajectory_index as i32;

        let ams_weight = trajectory.ams_weight;
        let mut rank_in_mother_branch: i32 = 0;

        let branch_size = trajectory.get_nb_branches();
        for branch_index in 0..branch_size {
            let mut cumulated_score = 0.;
            let mut last_branch_in_genealogy = true;
            let mut current_branch_index: i32 = branch_index as i32;
            #[allow(unused_assignments)]
            let mut last_point_index: i32 = 0;

            loop {
                let trajectory = &trajectories[current_trajectory_index as usize];
                let branch = trajectory.branches[current_branch_index as usize]
                    .lock()
                    .unwrap();

                if last_branch_in_genealogy {
                    last_point_index = (branch.points.len() - 1) as i32;
                } else {
                    last_point_index = rank_in_mother_branch;
                }

                for point_index in (0..last_point_index).rev() {
                    let point = branch.points[point_index as usize].clone();
                    let p = point.lock().unwrap();
                    let weight = p.weight;
                    let energy = p.energy;
                    let position = p.position;
                    let direction = p.direction;
                    let last_interaction = p.last_interaction;
                    let contribution = p.contribution;
                    drop(p);

                    if last_interaction == Interaction::ElasticScattering
                        || last_interaction == Interaction::Split
                        || last_interaction == Interaction::Source
                        || last_interaction == Interaction::Reflection
                    {
                        let scoring_point = Point {
                            position: position,
                            direction: direction,
                            energy: energy,
                            time: 0.,
                            weight: weight * ams_weight,
                            importance: 0.,
                            contribution: 0.,
                            cumulated_score: cumulated_score,
                            is_collision: false,
                            is_absorbed: false,
                            is_crossing: false,
                            last_interaction: last_interaction,
                        };
                        scoring_points.push(scoring_point.clone());

                        if importance_map.borrow().is_some(){
                            importance_map.borrow_mut().score(
                                Arc::new(Mutex::new(scoring_point)));
                        }
                    }

                    if last_interaction == Interaction::Jump
                        || last_interaction == Interaction::Cross
                        || last_interaction == Interaction::Absorption
                    {
                        cumulated_score += contribution * ams_weight;
                    }
                }
                if branch.mother_track == -1 {
                    break;
                }

                current_trajectory_index = branch.mother_track;
                current_branch_index = branch.mother_branch;
                rank_in_mother_branch = branch.rank_in_mother_branch;
                last_branch_in_genealogy = false;
            }
        }
    }
    if importance_map.borrow().is_some(){
        importance_map.borrow_mut().train();
    }
    scoring_points
}
