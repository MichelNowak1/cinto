use crate::common::cross_section_library::CrossSectionLibrary;
use crate::common::cross_section_library::Interaction;
use crate::common::geometry;
use crate::common::geometry::Geometry;
use crate::common::geometry::BoundaryCondition;
use crate::common::isotope::Isotope;
use crate::common::utils::get_index;
use crate::common::utils::weighted_sample;
use crate::common::importance_map::importance::Importance;

use rand::Rng;
use std::f64;
use std::cmp;
use std::sync::Arc;

use ndarray::prelude::*;

use pyo3::prelude::*;

/// Particle
///
/// # Attributes
/// * `position` position of the particle
/// * `direction` direction of the particle
/// * `energy` energy of the particle
/// * `weight` statistical weight of the particle
/// * `importance` importance of the particle
/// * `contribution` contribution of the particle
#[pyclass]
#[derive(Copy, Clone)]
pub struct Particle {
    #[pyo3(get, set)]
    pub position: [f64; 3],
    #[pyo3(get, set)]
    pub direction: [f64; 3],
    #[pyo3(get, set)]
    pub energy: f64,
    #[pyo3(get, set)]
    pub time: f64,
    #[pyo3(get, set)]
    pub weight: f64,

    #[pyo3(get, set)]
    pub importance: f64,
    pub contribution: f64,
    #[pyo3(get, set)]
    pub is_colliding: bool,
    pub is_absorbed: bool,
    pub is_crossing: bool,
    pub mother_trajectory_id: i32,
    pub mother_branch_id: i32,
    pub rank_in_mother_branch: i32,
    pub last_interaction: Interaction,
    pub initial_energy: f64,
}


#[pymethods]
impl Particle {
    /// Creates a particle
    ///
    /// # Returns
    /// * `particle` a particle.
    ///
    /// # Exampless
    /// ```
    /// use cinto::monte_carlo::particle::Particle;
    ///
    /// let particle = Particle::new();
    /// ```
    #[new]
    pub fn new() -> Self {
        Self {
            position: [0., 0., 0.],
            direction: [1., 0., 0.],
            energy: 14.,
            time: 0.,
            weight: 1.,
            importance: -1.,
            contribution: 0.,
            is_colliding: false,
            is_absorbed: false,
            is_crossing: false,
            mother_trajectory_id: -1,
            mother_branch_id: -1,
            rank_in_mother_branch: -1,
            last_interaction: Interaction::Null,
            initial_energy: 14.,
        }
    }

    /// Jumps to a new location in geometry
    ///
    /// # Arguments:
    /// * `geometry` geometry in which the particle is located
    /// * `cross_section_library` CrossSectionLibrary that needs to be used
    ///
    /// # Returns:
    /// * `done` boolean to know if transport ends or continues
    ///
    /// # Examples
    /// ```
    /// use cinto::common::source::Source;
    /// use cinto::common::geometry::Geometry;
    /// use cinto::common::cross_section_library::CrossSectionLibrary;
    ///
    /// let source = PunctualSource([0., 0., 0.]);
    ///
    /// let geometry = Geometry::new("test_geometry.root");
    ///
    /// let cross_section_library = CrossSectionLibrary::new("multigroup")
    ///                                                 .read_from_file("xs_lib.json");
    ///
    /// let particle = source.sample();
    ///
    /// particle.jump(&geometry, &cross_section_library);
    /// ```
    pub fn jump(
        &mut self,
        geometry: &geometry::Geometry,
        cross_section_library: &CrossSectionLibrary,
    ) -> bool {

        let mut done = false;
        self.is_crossing = false;
        self.is_colliding = false;

        let total_xs = geometry.get_material(self.position).get_sigma(
            Interaction::Total,
            self.energy,
            cross_section_library,
        );
        let mut rng = rand::thread_rng();
        let xsi: f64 = rng.r#gen();
        let jump_length: f64 = -1. / total_xs * xsi.ln();

        let dist_to_boundary: f64 = 
            geometry.get_distance_to_next_boundary(
                self.position,
                self.direction);

        let jump = jump_length.min(dist_to_boundary);

        self.position[0] += jump * self.direction[0];
        self.position[1] += jump * self.direction[1];
        self.position[2] += jump * self.direction[2];

        geometry.step_and_locate(jump_length);

        if jump < dist_to_boundary {
            self.is_colliding = true;
            self.last_interaction = Interaction::Jump;
            return false;
        }

        let geometry_precision = 1E-8;
        self.position[0] += geometry_precision * self.direction[0];
        self.position[1] += geometry_precision * self.direction[1];
        self.position[2] += geometry_precision * self.direction[2];

        self.last_interaction = Interaction::Cross;

        if geometry.get_volume_name(self.position) == "detector" {
            self.importance = std::f64::MAX;
        }

        if geometry.is_outside_of_geometry(self.position){
            return self.treat_boundary_conditions(geometry);
        }
        false
    }

    /// Collide on nucleus
    ///
    /// # Arguments:
    /// * `geometry` geometry in which the particle is located
    /// * `cross_section_library` CrossSectionLibrary that needs to be used
    /// * `implicite_capture` if true, implicite capture will be used
    ///
    /// # Returns:
    /// * `done` boolean to know if transport ends or continues
    ///
    /// # Examples
    /// ```
    /// particle.(&geometry, &cross_section_library, false);
    /// ```
    pub fn collide(
        &mut self,
        geometry: &geometry::Geometry,
        cross_section_library: &CrossSectionLibrary,
        implicite_capture: bool,
    ) -> (bool, Vec<Particle>) {
        self.is_absorbed = false;
        self.initial_energy = self.energy;

        let produced_particles = Vec::<Particle>::new();

        // get material at particle position
        let material = geometry.get_material(self.position);

        // sample isotope on which to interact at particle energy
        let isotope = material.sample_isotope(
            self.energy,
            cross_section_library);

        // sample the interaction on the isotope at particle energy
        let mut interaction = isotope.sample_interaction(self.energy);

        if implicite_capture && interaction == Interaction::Absorption {
            interaction = Interaction::ElasticScattering;

            self.weight *= 1. -
                isotope.get_sigma_abs(self.energy) /
                isotope.get_sigma_tot(self.energy);
        }

        let done = match interaction {
            Interaction::ElasticScattering => self.elastic_scatter(isotope),
            Interaction::Absorption => self.absorb(),
            Interaction::InelasticScattering => panic!("Inelastic scattering not implemented yet"),
            Interaction::Total => panic!("This should not happen."),

            Interaction::Fission => panic!("not implementented"),
            Interaction::Scattering => panic!("this should not happen"),
            Interaction::Null => panic!("this should not happen"),
            Interaction::Source => panic!("this should not happen"),
            Interaction::Split => panic!("this should not happen"),
            Interaction::Reflection => panic!("this should not happen"),
            Interaction::Jump => panic!("this should not happen"),
            Interaction::Cross => panic!("this should not happen"),
            Interaction::Leak => panic!("this should not happen"),
        };

        self.last_interaction = interaction;

        (done, produced_particles)
    }
}

impl Particle {

    /// Treats boundary conditions
    ///
    /// # Returns
    /// * `done` true if the transport needs to be continued
    ///
    /// # Exampless
    /// ```
    /// use cinto::monte_carlo::particle::Particle;
    ///
    /// let particle = Particle::new();
    /// ```
    pub fn treat_boundary_conditions(
        &mut self,
        geometry: &Geometry
        ) -> bool {

        let boundary_condition = geometry.get_boundary_condition(
            self.position,
            self.direction);

        if !boundary_condition.is_some() {
            self.is_crossing = true;
            self.last_interaction = Interaction::Cross;
            return false;
        }

        match boundary_condition.unwrap(){

            BoundaryCondition::Leak=> {
                self.last_interaction = Interaction::Leak;
                true
            },
            BoundaryCondition::Reflective => {
                let normal = geometry.get_normal(
                    self.position,
                    self.direction);

                let direction_normal_dot_product =
                      self.direction[0] * normal[0]
                    + self.direction[1] * normal[1]
                    + self.direction[2] * normal[2];

                self.direction[0] -= 2. * direction_normal_dot_product * normal[0];
                self.direction[1] -= 2. * direction_normal_dot_product * normal[1];
                self.direction[2] -= 2. * direction_normal_dot_product * normal[2];
                geometry.set_direction(self.direction);

                self.last_interaction = Interaction::Reflection;

                false
            }
        }
    }

    /// Absorbs the particle
    ///
    /// # Returns:
    /// * `done` boolean to know if transport ends or continues
    ///
    /// # Examples
    /// ```
    /// use cinto::common::source::Source;
    /// use cinto::common::geometry::Geometry;
    /// use cinto::common::cross_section_library::CrossSectionLibrary;
    ///
    /// let source = PunctualSource([0., 0., 0.]);
    ///
    /// let geometry = Geometry::new("test_geometry.root");
    ///
    /// let cross_section_library = CrossSectionLibrary::new("multigroup")
    ///                                                 .read_from_file("xs_lib.json");
    ///
    /// let particle = source.sample();
    /// particle.biased_jump(&geometry, &cross_section_library, &importance_map);
    /// ```
    pub fn absorb(&mut self) -> bool {
        self.is_absorbed = true;
        self.importance = std::f64::MIN;
        true
    }

    /// Scatters the particle elastically.
    ///
    /// # Arguments:
    /// `isotope` isotope on which the scattering is performed.
    ///
    /// # Returns:
    /// * `done` boolean to know if transport ends or continues
    ///
    /// # Examples
    /// ```
    /// use cinto::common::geometry::Geometry;
    /// use cinto::common::cross_section_library::CrossSectionLibrary;
    ///
    /// let geometry = Geometry::new("test_geometry.root");
    ///
    /// let cross_section_library = CrossSectionLibrary::new("multigroup")
    ///                                                 .read_from_file("xs_lib.json");
    /// let position = [0., 0., 0.];
    /// let direction = [0., 1., 0.];
    /// let energy = 14.; // MeV
    ///
    /// let particle = Particle::new(position, direction, energy);
    /// ```
    pub fn elastic_scatter(&mut self, isotope: Arc<Isotope>) -> bool {
        // is scattering matrix is present, then calculation is multigroup
        if isotope.scattering_matrix.is_some() {
            self.multigroup_scatter(isotope);
        }
        // else, the representation is punctual
        else {
            self.punctual_scatter(isotope);
        }
        false
    }

    /// Scatters the particle with the multigroup formalism.
    ///
    /// # Arguments:
    /// `isotope` isotope on which the scattering is performed.
    ///
    ///
    /// # Examples
    /// ```
    /// let position = [0., 0., 0.];
    /// let direction = [0., 1., 0.];
    /// let energy = 14.; // MeV
    ///
    /// let particle = Particle::new(position, direction, energy);
    ///
    /// particle.multigroup_scatter(isotope: Isotope);
    /// ```
    pub fn multigroup_scatter(&mut self, isotope: Arc<Isotope>) {

        let index = get_index(self.energy, &isotope.energies);
        if !index.is_some() {
            panic!("multigroup scatter at energy {} impossible", self.energy);
        }
        let incoming_energy_slot = index.unwrap();
        let scattering_probabilities = isotope
            .scattering_matrix
            .as_ref()
            .unwrap()
            .slice(s![.., incoming_energy_slot]);

        let output_energy_slot = weighted_sample(&scattering_probabilities.to_vec());

        // sample energy uniformly in group
        let mut rng = rand::thread_rng();
        let xsi: f64 = rng.r#gen();

        let up_out = isotope.energies[output_energy_slot];
        let low_out = isotope.energies[output_energy_slot + 1];

        let output_energy = low_out + xsi * (up_out - low_out);
        // let output_energy = (up_out + low_out)/2.;

        self.initial_energy = self.energy;
        self.energy = output_energy;

        self.change_direction_isotropically();
    }

    /// Changes the direction with given cos_theta.
    ///
    /// # Arguments:
    /// `cos_theta` cosine of angle of deviation with which the scattering needs to be performed.
    ///
    /// # Examples
    /// ```
    /// let position = [0., 0., 0.];
    /// let direction = [0., 1., 0.];
    /// let energy = 14.; // MeV
    ///
    /// let particle = Particle::new(position, direction, energy);
    ///
    /// particle.multigroup_scatter(isotope: Isotope);
    /// ```
    pub fn change_direction_anisotropically(&mut self, cos_theta: f64) {
        // sample \phi
        let mut rng = rand::thread_rng();
        let xsi_phi: f64 = rng.r#gen();
        let phi = 2. * std::f64::consts::PI * xsi_phi;

        // convert \theta and \phi in cartesian coordinates
        self.direction[0] = (1. - cos_theta * cos_theta).sqrt() * phi.cos();
        self.direction[1] = (1. - cos_theta * cos_theta).sqrt() * phi.sin();
        self.direction[2] = cos_theta;
    }

    /// Changes the direction isotropically
    ///
    /// # Arguments:
    /// `cos_theta` cosine of angle of deviation with which the scattering needs to be performed.
    ///
    /// # Examples
    /// ```
    /// let position = [0., 0., 0.];
    /// let direction = [0., 1., 0.];
    /// let energy = 14.; // MeV
    ///
    /// let particle = Particle::new(position, direction, energy);
    ///
    /// particle.multigroup_scatter(isotope: Isotope);
    /// ```
    pub fn change_direction_isotropically(&mut self) {
        // d^2S = r^2 d\theta \sin(\theta) d\phi
        // isotropic condition: \int d^2S = 1
        // \phi \in  [0,2\pi]
        // \theta \in  [0,\pi]
        // so phi sampled uniformly in [0,2\pi]
        // and \cos(\theta) sampled uniformy from -1 to 1

        // get random number generator
        let mut rng = rand::thread_rng();

        // sample \theta
        let xsi_cos_theta: f64 = rng.r#gen();
        let cos_theta = -1. + 2. * xsi_cos_theta;

        // sample \phi
        let xsi_phi: f64 = rng.r#gen();
        let phi = 2. * std::f64::consts::PI * xsi_phi;

        // convert \theta and \phi in cartesian coordinates
        self.direction[0] = (1. - cos_theta * cos_theta).sqrt() * phi.cos();
        self.direction[1] = (1. - cos_theta * cos_theta).sqrt() * phi.sin();
        self.direction[2] = cos_theta;
    }

    /// Scatters the particle in the continuous energy formalism.
    ///
    /// # Arguments:
    /// `cos_theta` cosine of angle of deviation with which the scattering needs to be performed.
    ///
    /// # Examples
    /// ```
    /// let position = [0., 0., 0.];
    /// let direction = [0., 1., 0.];
    /// let energy = 14.; // MeV
    ///
    /// let particle = Particle::new(position, direction, energy);
    ///
    /// particle.multigroup_scatter(isotope: Isotope);
    /// ```
    pub fn punctual_scatter(&mut self, isotope: Arc<Isotope>) {

        // Compute outgoing energy
        // Applied Reactor Physics: Alain Hébert, Chapter 2. Dynamics of scattering reactions
        // \alpha = ((A-1)/(A+1))^2
        // uniform distribution of outgoing energy between \alpha E_in and E_in

        // get nucleid mass
        let nucleid_mass = isotope.atomic_mass;

        // compute alpha
        let alpha = ((nucleid_mass - 1.) / (nucleid_mass + 1.)).powi(2);

        // sample random number
        let mut rng = rand::thread_rng();
        let xsi_energy: f64 = rng.r#gen();

        let initial_energy = self.energy;
        self.energy = alpha * self.energy + xsi_energy * (self.energy - alpha * self.energy);

        let cos_theta = 0.5 * (nucleid_mass + 1.) * (self.energy / initial_energy).sqrt()
            - 0.5 * (nucleid_mass - 1.) * (initial_energy / self.energy).sqrt();

        self.change_direction_anisotropically(cos_theta);
    }
}
