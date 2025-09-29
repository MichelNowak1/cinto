use crate::common::material::Material;
use std::collections::HashMap;
use std::f64;

use libc::c_char;
use std::ffi::{CStr, CString};

use pyo3::prelude::*;

#[link(name = "cinto_root_geometry", kind = "static")]
unsafe extern "C" {
    fn init_root_geometry(file_name: *const c_char);
    fn get_geometry_coordinates() -> *const f64;
    fn get_volume_name(position: *const f64) -> *const c_char;
    fn get_volume_capacity(position: *const f64) -> f64;
    fn get_distance_to_next_boundary(position: *const f64, direction: *const f64) -> f64;
    fn is_outside_of_geometry(position: *const f64) -> bool;
    fn set_position_and_direction(position: *const f64, direction: *const f64);
    fn set_direction(direction: *const f64);
    fn get_normal_on_boundary(position: *const f64 , direction: *const f64)->  *const f64;
    fn is_on_boundary(thread_id: u64) -> bool;
    fn step_and_locate(jump_length: f64) -> *const f64;
    fn cross_boundary_and_locate(jump_length: f64) -> *const f64;
    fn init_thread();
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum BoundaryCondition{
    Reflective,
    Leak,
}

/// Geometry
///
/// # Arguments
/// * `root_file_name` name of file containing the root geometry
/// * `materials` list of materials present in the geometry
/// * `volume_material_assosiation` map of material names to
/// * `boundary_conditions` array of BoundaryConditions [[x_min_condition, x_max_condition],
///                                                      [y_min_condition, y_max_condition],
///                                                      [z_min_condition, z_max_condition]]
/// * `bounds` array of bounds coordinates [[x_min, x_max],
///                                         [y_min, y_max],
///                                         [z_min, z_max]]
///            0: vacuum, 1: reflective
#[pyclass]
pub struct Geometry {
    pub root_file_name: String,
    #[pyo3(get, set)]
    pub materials: Vec<Material>,
    pub volume_material_association: HashMap<String, String>,
    pub boundary_conditions: [[BoundaryCondition; 2]; 3],
    pub bounds: [[f64; 2]; 3],
}

#[pymethods]
impl Geometry {
    /// Build of geometry
    ///
    /// # Arguments
    /// * `root_file_name` name of the file containing the root geometry
    ///
    /// # Returns
    /// * `geometry` a new geometry
    ///
    /// # Examples
    /// ```
    /// use cinto::common::geometry::Geometry;
    ///
    /// let geometry = Geometry::new("test_geometry.root".to_string());
    /// ```
    #[new]
    pub fn new(root_file_name: String) -> Self {
        Self {
            root_file_name: root_file_name,
            materials: Vec::new(),
            volume_material_association: HashMap::new(),
            boundary_conditions: [
                 [BoundaryCondition::Leak, BoundaryCondition::Leak],
                 [BoundaryCondition::Leak, BoundaryCondition::Leak],
                 [BoundaryCondition::Leak, BoundaryCondition::Leak]],
            bounds: [[0., 0.]; 3],
        }
    }

    /// Initialize the geometry
    ///
    /// # Arguments
    /// * `material_name` name of material to be added to the geometry
    ///
    /// # Examples
    /// ```
    /// use cinto::common::geometry::Geometry;
    ///
    /// let geometry = Geometry::new("test_geometry.root".to_string());
    /// geometry.add_material("Water (Liquid");
    /// ```
    pub fn init(&mut self) {
        let _file_str = CString::new(self.root_file_name.to_string()).expect("CString::new failed");
        unsafe {
            init_root_geometry(_file_str.as_ptr());
        }
        let coordinates = self.get_geometry_coordinates();
        self.bounds[0][0] = coordinates[0];
        self.bounds[0][1] = coordinates[1];
        self.bounds[1][0] = coordinates[2];
        self.bounds[1][1] = coordinates[3];
        self.bounds[2][0] = coordinates[4];
        self.bounds[2][1] = coordinates[5];
    }

    /// Add a material to the geometry
    ///
    /// # Arguments
    /// * `material_name` name of material to be added to the geometry
    ///
    /// # Examples
    /// ```
    /// use cinto::common::geometry::Geometry;
    ///
    /// let geometry = Geometry::new("test_geometry.root".to_string());
    /// geometry.add_material("Water (Liquid");
    /// ```
    pub fn add_material(&mut self, material_name: String) {
        // let material = Material::new(material_name);
        let mat = Material {
            name: material_name.clone(),
            density: 1.0,
            isotope_names: vec![material_name.clone()],
            isotope_concentrations: vec![1.0],
        };
        self.materials.push(mat);
    }

    pub fn set_volume_material_association(&mut self, volume_material_association: HashMap<String, String>) {
        self.volume_material_association = volume_material_association;
    }

    pub fn init_thread(&self) {
        unsafe {
            init_thread();
        }
    }

    pub fn set_boundary_conditions(&mut self, boundary_conditions: Vec<String>) {
        for (i, bc_dim) in boundary_conditions.iter().enumerate() {
            if bc_dim.eq("reflective") {
                self.boundary_conditions[i / 2][i % 2] = BoundaryCondition::Reflective;
            }
        }
    }
}

impl Geometry {

    /// Get volume name at a given position
    ///
    /// # Arguments
    /// * `position` position which we wish to get the volume name
    ///
    /// # Returns
    /// * `volume_name` retrieves the volume name at provided position
    ///
    /// # Examples
    /// ```
    /// use cinto::common::geometry::Geometry;
    ///
    /// let geometry = Geometry::new("test_geometry.root".to_string());
    /// let position = [0., 0., 0.];
    /// let volume_name = geometry.get_volume_name(position);
    /// ```
    pub fn get_volume_name(&self, position: [f64; 3]) -> String {
        let raw_position = &position as *const f64;

        unsafe {
            let vn = get_volume_name(raw_position);
            let c_str: &CStr = CStr::from_ptr(vn);
            c_str.to_str().unwrap().to_string().to_owned()
        }
    }

    /// Get volume name at a given position
    ///
    /// # Arguments
    /// * `position` position which we wish to get the volume name
    ///
    /// # Returns
    /// * `volume_name` retrieves the volume name at provided position
    ///
    /// # Examples
    /// ```
    /// use cinto::common::geometry::Geometry;
    ///
    /// let geometry = Geometry::new("test_geometry.root".to_string());
    /// let position = [0., 0., 0.];
    /// let volume_name = geometry.get_volume_capacity(position);
    /// ```
    pub fn get_volume_capacity(&self, position: [f64; 3]) -> f64 {
        let raw_position = &position as *const f64;

        unsafe { get_volume_capacity(raw_position) }
    }

    /// Get material
    ///
    /// # Arguments
    /// * `position` position which we wish to get material
    ///
    /// # Returns
    /// * `material` a reference to the material
    ///
    /// # Examples
    /// ```
    /// use cinto::common::geometry::Geometry;
    ///
    /// let geometry = Geometry::new("test_geometry.root".to_string());
    /// let position = [0., 0., 0.];
    /// let material = geometry.get_material(position);
    /// ```
    pub fn get_material(&self, position: [f64; 3]) -> &Material {
        let volume_name = self.get_volume_name(position);
        let material_name = &self.volume_material_association[&volume_name];
        let mut retrieved_index = 0;
        for (index, material) in self.materials.iter().enumerate() {
            if material.name.eq(material_name) {
                retrieved_index = index;
            }
        }
        &self.materials[retrieved_index]
    }

    /// Get distance to next boundary
    ///
    /// # Arguments
    /// * `position` position of the particle
    /// * `direction` direction of the particle
    ///
    /// # Returns
    /// * `distance` the distance to the next boundary
    ///
    /// # Examples
    /// ```
    /// use cinto::common::geometry::Geometry;
    ///
    /// let geometry = Geometry::new("test_geometry.root".to_string());
    ///
    /// let position = [0., 0., 0.];
    /// let direction = [1., 0., 0.];
    /// let distance = geometry.get_distance_to_next_boundary(position, direction);
    /// ```
    pub fn get_distance_to_next_boundary(&self, position: [f64; 3], direction: [f64; 3]) -> f64 {
        let raw_position = &position as *const f64;
        let raw_direction = &direction as *const f64;

        #[allow(unused_assignments)]
        let mut distance: f64 = 0.;
        unsafe { distance = get_distance_to_next_boundary(raw_position, raw_direction) }
        distance
    }

    /// Returns true if point is outside of geometry
    ///
    /// # Arguments
    /// * `position` position of the particle
    ///
    /// # Returns
    /// * `outside` true if outside, false otherwise
    ///
    /// # Examples
    /// ```
    /// use cinto::common::geometry::Geometry;
    ///
    /// let geometry = Geometry::new("test_geometry.root".to_string());
    ///
    /// let position = [0., 0., 0.];
    /// let outside = geometry.is_outside_of_geometry(position);
    /// ```
    pub fn is_outside_of_geometry(&self, position: [f64; 3]) -> bool {
        if position[0] < self.bounds[0][0] ||
           position[0] > self.bounds[0][1] ||
           position[1] < self.bounds[1][0] ||
           position[1] > self.bounds[1][1] ||
           position[2] < self.bounds[2][0] ||
           position[2] > self.bounds[2][1] {
            return true;
           }
        return false;

        // TODO implememnt this with ROOT
        // let raw_position = &position as *const f64;
        // #[allow(unused_assignments)]
        // let mut outside: bool = false;
        // unsafe { outside = is_outside_of_geometry(raw_position) }
        // outside
    }

    /// Returns the cartesian bounds of the geometry
    ///
    /// # Returns
    /// * `array` x_min, x_max, y_min, y_max, z_max
    ///
    /// # Examples
    /// ```
    /// use cinto::common::geometry::Geometry;
    ///
    /// let geometry = Geometry::new("test_geometry.root".to_string());
    ///
    /// let geometry_coordinates = geometry.get_geometry_coordinates();
    /// ```
    pub fn get_geometry_coordinates(&self) -> [f64; 6] {
        let c = unsafe { std::slice::from_raw_parts(get_geometry_coordinates(), 6) };
        [c[0], c[1], c[2], c[3], c[4], c[5]]
    }

    /// Sets the position int the root C++ side
    ///
    /// # Arguments
    /// * `position` position of the particle
    /// * `direction` direction of the particle
    ///
    /// # Examples
    /// ```
    /// use cinto::common::geometry::Geometry;
    ///
    /// let geometry = Geometry::new("test_geometry.root".to_string());
    ///
    /// let position = [0., 0., 0.];
    /// let direction = [1., 0., 0.];
    ///
    /// geometry.set_position_and_direction(position, direction);
    /// ```
    pub fn set_position_and_direction(
        &self,
        position: [f64; 3],
        direction: [f64; 3]) {
        let raw_position = &position as *const f64;
        let raw_direction = &direction as *const f64;
        unsafe { set_position_and_direction(raw_position, raw_direction) }
    }

    pub fn set_direction(&self, direction: [f64; 3]) {
        let raw_direction = &direction as *const f64;
        unsafe { set_direction(raw_direction) }
    }

    pub fn get_normal(
        &self,
        position: [f64; 3],
        direction: [f64; 3]
        ) -> [f64; 3]{

        let raw_position = &position as *const f64;
        let raw_direction = &direction as *const f64;

        let normal = unsafe {
            std::slice::from_raw_parts(
                get_normal_on_boundary(
                    raw_position,
                    raw_direction), 3)};
        [normal[0], normal[1], normal[2]]
    }

    pub fn get_boundary_condition(
        &self,
        position: [f64; 3],
        direction: [f64; 3]
        ) -> Option<BoundaryCondition> {

        for direction in 0..3 {
            for side in 0..2 {

                let distance_to_plane = (
                    self.bounds[direction][side] -
                    position[direction]).abs();

                if distance_to_plane < 1E-6 {
                    return Some(self.boundary_conditions[direction][side])
                }
            }
        }
        None
    }
    pub fn step_and_locate(
        &self,
        jump_length: f64) -> [f64 ;3]{

        let current_position = unsafe{
            std::slice::from_raw_parts(
                step_and_locate(jump_length), 3)
        };
        [current_position[0],
         current_position[1],
         current_position[2]]
    }
    pub fn cross_boundary_and_locate(
        &self,
        jump_length: f64) -> [f64 ;3]{
        let current_position = unsafe{
            std::slice::from_raw_parts(
                cross_boundary_and_locate(jump_length), 3)
        };
        [current_position[0],
         current_position[1],
         current_position[2]]
    }
}
