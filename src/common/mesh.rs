use crate::common::cross_section_library::CrossSectionLibrary;
use crate::monte_carlo::trajectory::Trajectory;
use crate::monte_carlo::trajectory::Point;
use crate::common::geometry::Geometry;

pub enum MeshType {
    Cartesian,
    Spherical,
}
pub trait Mesh {
    fn get_index(&self, point: &Point) -> Option<[usize; 4]>;
    fn add_tally(
        &mut self,
        trajectory: &Trajectory,
        geometry: &Geometry,
        cross_section_library: &CrossSectionLibrary,
    );
    fn normalize(&mut self, norm: f64);
    fn prepare_next_batch(&mut self);
    fn normalize_ams_weight(&mut self, ams_weight: f64);
    fn dump(&self);
}
