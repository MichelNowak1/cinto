extern crate rand;

use crate::common::cross_section_library::CrossSectionLibrary;
use crate::common::geometry::Geometry;

use crate::monte_carlo::particle;


/// Transport particle
///
/// # Arguments
/// * `particle` particle to be transported
/// * `geometry` geometry in which the particle needs to be transported
/// * `detectors` list of detectors to be scored during transport
/// * `importance_map` importance_map to be used for variance reduction techniques
/// * `importance_sampling` boolean flag, if true, importance sampling is used for transport
///
/// # Returns
/// * `produced_particles` particles that have been produced during transport
///
/// # Example
/// ```
/// use crate cinto::transport::transport;
/// let particle = source.sample();
/// let produced_particles = transport(particle,
///                                    geometry,
///                                    cross_section_library,
///                                    detectors);
/// ```
pub fn step_flight(
    particle: &mut particle::Particle,
    geometry: &Geometry,
    cross_section_library: &CrossSectionLibrary,
) -> bool {
    particle.jump(&geometry, &cross_section_library)
}

pub fn step_collision(
    particle: &mut particle::Particle,
    geometry: &Geometry,
    cross_section_library: &CrossSectionLibrary,
) -> (bool, Vec<particle::Particle>) {

    let mut produced_particles = Vec::<particle::Particle>::new();

    let (continue_transport, mut _produced_particles) = particle.collide(
        &geometry,
        &cross_section_library,
        false,
    );
    produced_particles.append(&mut _produced_particles);
    if particle.energy <= 1E-11 {
        return (false, produced_particles);
    }

    (continue_transport, produced_particles)
}
