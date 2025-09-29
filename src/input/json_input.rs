extern crate libc;
use std::fs::File;

use crate::common::cartesian_mesh;
use crate::common::cross_section_library::CrossSectionLibrary;
use crate::common::cross_section_library::EvaluationType;
use crate::common::cross_section_library::Interaction;
use crate::common::detector;
use crate::common::geometry;
use crate::common::geometry::BoundaryCondition;
use crate::common::material::Material;
use crate::common::mesh::Mesh;
use crate::common::source;
use crate::common::spherical_mesh;
use crate::common::importance_map;
use crate::common::importance_map::importance::Importance;

use crate::monte_carlo::homogenizer;
use crate::monte_carlo::strategy;
use crate::monte_carlo::tally::Tally;

use ndarray::prelude::*;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

pub struct JSONInput {
    pub dict: serde_json::Value,
}

impl JSONInput {
    pub fn new(file_path: String) -> JSONInput {
        let file = File::open(file_path).expect("file should open read only");
        JSONInput {
            dict: serde_json::from_reader(file).expect("file should be proper JSON"),
        }
    }
    pub fn get_output_file_name(&self) -> String {
        let _name = self.dict["output_file"].as_str().unwrap();
        _name.to_string()
    }
    pub fn get_name(&self) -> String {
        let _name = self.dict["name"].as_str().unwrap();
        _name.to_string()
    }
    pub fn get_calculation_type(&self) -> String{
        if self.dict["calculation_type"] == serde_json::Value::Null {
            panic!("Define a calculation_type for the simulation to run.")
        }
        
        self.dict["calculation_type"].as_str().unwrap().to_string()
    }
    pub fn get_geometry(&self, cross_section_library: &mut CrossSectionLibrary) -> geometry::Geometry {
        let mut materials: Vec<Material> = Vec::new();
        let mut volume_material_association = HashMap::<String, String>::new();

        if self.dict["geometry"] == serde_json::Value::Null {
            panic!("Define a geometry for the simulation to run.")
        }
        let geometry_dict = &self.dict["geometry"];
        // retrieve name of root file
        let geometry_file = geometry_dict["root_file_name"].as_str().unwrap();

        // retrieve material list
        let material_list = &geometry_dict["material_list"];

        for material_dict in material_list.as_array().unwrap() {
            // use pyne
            if material_dict.as_str().is_some() {
                let material = Material::new(material_dict.as_str().unwrap().to_string());
                materials.push(material);
            } else {
                // use local cross section library

                let material_name = material_dict["name"].as_str().unwrap().to_string();

                let isotopes = material_dict["isotopes"].as_array().unwrap();
                let concentrations = material_dict["concentrations"].as_array().unwrap();

                let mut material = Material {
                    name: material_name,
                    density: 1.0,
                    isotope_names: Vec::new(),
                    isotope_concentrations: Vec::new(),
                };

                for (isotope, concentration) in isotopes.iter().zip(concentrations) {
                    let iso = isotope.as_str()
                                     .unwrap()
                                     .to_string();
                    material.isotope_names.push(iso);
                    material
                        .isotope_concentrations
                        .push(concentration.as_f64().unwrap());
                }
                materials.push(material);
            }
        }

        // retrieve association list of volumes and materials
        let vm_list = &geometry_dict["volume_material"];
        for vm in vm_list.as_array().unwrap() {
            volume_material_association.insert(
                vm["volume_name"].as_str().unwrap().to_string(),
                vm["material_name"].as_str().unwrap().to_string(),
            );
        }
        let mut boundary_conditions = [
                 [BoundaryCondition::Leak, BoundaryCondition::Leak],
                 [BoundaryCondition::Leak, BoundaryCondition::Leak],
                 [BoundaryCondition::Leak, BoundaryCondition::Leak]];

        if geometry_dict["boundary_conditions"] != serde_json::Value::Null {
            // min boundary conditions
            if geometry_dict["boundary_conditions"]["x_min"] == "reflective" {
                boundary_conditions[0][0] = BoundaryCondition::Reflective;
            }
            if geometry_dict["boundary_conditions"]["y_min"] == "reflective" {
                boundary_conditions[1][0] = BoundaryCondition::Reflective;
            }
            if geometry_dict["boundary_conditions"]["z_min"] == "reflective" {
                boundary_conditions[2][0] = BoundaryCondition::Reflective;
            }
            // max boundary conditions
            if geometry_dict["boundary_conditions"]["x_max"] == "reflective" {
                boundary_conditions[0][1] = BoundaryCondition::Reflective;
            }
            if geometry_dict["boundary_conditions"]["y_max"] == "reflective" {
                boundary_conditions[1][1] = BoundaryCondition::Reflective;
            }
            if geometry_dict["boundary_conditions"]["z_max"] == "reflective" {
                boundary_conditions[2][1] = BoundaryCondition::Reflective;
            }
        }

        let bounds = [[0., 1.], [0., 1.], [0., 1.]];

        // Create geometry
        let geometry = geometry::Geometry {
            root_file_name: geometry_file.to_string(),
            materials: materials,
            volume_material_association: volume_material_association,
            boundary_conditions: boundary_conditions,
            bounds: bounds,
        };
        geometry
    }
    pub fn get_detector(&self) -> Option<detector::Detector> {
        let _detector_dict = &self.dict["detector"];
        if self.dict["detector"] == serde_json::Value::Null {
            return None;
        }
        let _volume_name = _detector_dict["volume_name"].as_str().unwrap().to_string();
        let mut response_type = Interaction::Null;
        if _detector_dict["response_type"] == "scattering" {
            response_type = Interaction::ElasticScattering;
        }

        let _ebounds = _detector_dict["energy_bounds"].as_array().unwrap();
        let ebounds: Vec<f64> = _ebounds.iter().map(|f| f.as_f64().unwrap()).collect();

        let _detector = Some(detector::Detector {
            volume_name: _volume_name,
            response_type: response_type,
            tally: Tally::new(),
            importance: std::f64::MAX,
            scoring: true,
            energy_bounds: ebounds,
        });
        _detector
    }
    pub fn get_source(&self) -> Option<source::Source> {
        if self.dict["source"] == serde_json::Value::Null {
            panic!("Define a source for the simulation to run.")
        }
        let _source_dict = &self.dict["source"];
        let _source_type: String = _source_dict["type"].as_str().unwrap().to_string();

        if _source_type.eq("Punctual") {
            let mut direction: Vec<f64> = [0., 0., 0.].to_vec();
            let _position = _source_dict["parameters"]["position"].as_array().unwrap();
            let position: Vec<f64> = _position.iter().map(|f| f.as_f64().unwrap()).collect();
            let _energy: f64 = _source_dict["parameters"]["energy"].as_f64().unwrap();

            if _source_dict["parameters"]["direction"] != serde_json::Value::Null {
                let _direction = _source_dict["parameters"]["direction"].as_array().unwrap();
                direction = _direction.iter().map(|f| f.as_f64().unwrap()).collect();
            }

            let _source = source::Source {
                position: [position[0], position[1], position[2]],
                direction: [direction[0], direction[1], direction[2]],
                energy: _energy,
            };

            Some(_source)
        } else {
            None
        }
    }
    pub fn get_num_batches(&self) -> usize {
        let _num_batches = self.dict["monte_carlo"]["num_batches"].as_u64().unwrap() as usize;
        _num_batches
    }
    pub fn get_num_particles_per_batch(&self) -> usize {
        let mut batch_size = 0;
        if self.dict["monte_carlo"] != serde_json::Value::Null {
            batch_size = self.dict["monte_carlo"]["particles_per_batch"]
                .as_u64()
                .unwrap() as usize;
        }
        batch_size
    }
    pub fn get_importance_type(&self, map_name: String) -> String {
        let mut itype: String = "unknown".to_string();
        if self.dict["monte_carlo"]["variance_reduction"][&map_name] != serde_json::Value::Null {
            let _importance_dict = &self.dict["monte_carlo"]["variance_reduction"][&map_name];
            itype = _importance_dict["type"].as_str().unwrap().to_string();
        }
        itype
    }
    pub fn get_importance(&self, map_name: String) -> Rc<RefCell<dyn Importance>> {
        if self.dict["monte_carlo"]["variance_reduction"][&map_name] != serde_json::Value::Null {
            let _itype = self.get_importance_type(map_name.to_string());
            if _itype.eq("scalar_map") || _itype.eq("directional_map")
            {
                return self.get_importance_map(map_name.to_string(), _itype.to_string());
            } else if _itype.eq("attractor") || _itype.eq("repeller") {
                return Rc::new(RefCell::new(self.get_importance_function(&_itype)));
            } else if _itype.eq("learner") {
                panic!("left befing learner for now");
                // return Rc::new(RefCell::new(self.get_learner_importance(map_name.to_string())));
            }
        }
        Rc::new(RefCell::new(importance_map::null::Null {}))
    }
    fn get_importance_map(
        &self,
        map_name: String,
        map_type: String,
    ) -> Rc<RefCell<dyn Importance>> {
        let importance_dict = &self.dict["monte_carlo"]["variance_reduction"][&map_name];

        if importance_dict["from_file"] != serde_json::Value::Null {
            return Rc::new(RefCell::new(importance_map::scalar::ScalarImportanceMap::new(
                       importance_dict["from_file"].as_str()
                                           .unwrap()
                                           .to_string()
                                        )
                    )
               );
        }

        if map_type.eq("scalar_map") {
            let x_bounds: Vec<f64> = importance_dict["x_bounds"].as_array().unwrap().iter().map(|f| f.as_f64().unwrap()).collect();
            let y_bounds: Vec<f64> = importance_dict["y_bounds"].as_array().unwrap().iter().map(|f| f.as_f64().unwrap()).collect();
            let z_bounds: Vec<f64> = importance_dict["z_bounds"].as_array().unwrap().iter().map(|f| f.as_f64().unwrap()).collect();
            let energy_bounds: Vec<f64> = importance_dict["energy_bounds"].as_array().unwrap().iter().map(|f| f.as_f64().unwrap()).collect();
            return Rc::new(RefCell::new(importance_map::scalar::ScalarImportanceMap::build_new(
                x_bounds, y_bounds, z_bounds, energy_bounds)));
        } else if map_type.eq("directional_map") {
            let x_bounds = importance_dict["x_bounds"].as_array().unwrap().iter().map(|f| f.as_f64().unwrap()).collect();
            let y_bounds = importance_dict["y_bounds"].as_array().unwrap().iter().map(|f| f.as_f64().unwrap()).collect();
            let z_bounds = importance_dict["z_bounds"].as_array().unwrap().iter().map(|f| f.as_f64().unwrap()).collect();
            let energy_bounds = importance_dict["energy_bounds"].as_array().unwrap().iter().map(|f| f.as_f64().unwrap()).collect();
            let theta_bounds = importance_dict["theta_bounds"].as_array().unwrap().iter().map(|f| f.as_f64().unwrap()).collect();
            let phi_bounds = importance_dict["phi_bounds"].as_array().unwrap().iter().map(|f| f.as_f64().unwrap()).collect();
            return Rc::new(RefCell::new(importance_map::directional::DirectionalImportanceMap::new(
                x_bounds, y_bounds, z_bounds, energy_bounds, theta_bounds, phi_bounds)));
        } else {
            panic!(
                "Unkown importance map type,
                    try using scalar_map or directional_map"
            );
        }
    }
    fn get_importance_function(&self, imtype: &String) -> importance_map::functional::FunctionalImportance {
        let importance_dict = &self.dict["monte_carlo"]["variance_reduction"]["importance"];
        let origin = importance_dict["origin"].as_array().unwrap();
        let ifun = importance_map::functional::FunctionalImportance {
            origin: origin.iter().map(|f| f.as_f64().unwrap()).collect(),
            itype: imtype.to_string(),
        };
        ifun
    }

    pub fn get_mesh(&self) -> Option<Box<dyn Mesh>> {
        if self.dict["mesh"] == serde_json::Value::Null {
            return None;
        }
        let _mesh_dict = &self.dict["mesh"]["spatial_mesh"];
        let _mesh_type = &_mesh_dict["geometrical_type"];

        if _mesh_type.eq("cartesian") {
            let _x_bounds = _mesh_dict["x_bounds"].as_array().unwrap();
            let x_bounds: Vec<f64> = _x_bounds.iter().map(|f| f.as_f64().unwrap()).collect();

            let _y_bounds = _mesh_dict["y_bounds"].as_array().unwrap();
            let y_bounds: Vec<f64> = _y_bounds.iter().map(|f| f.as_f64().unwrap()).collect();

            let _z_bounds = _mesh_dict["z_bounds"].as_array().unwrap();
            let z_bounds: Vec<f64> = _z_bounds.iter().map(|f| f.as_f64().unwrap()).collect();

            let _energy_mesh = &self.dict["mesh"]["energy_bounds"].as_array().unwrap();
            let energy_mesh: Vec<f64> = _energy_mesh.iter().map(|f| f.as_f64().unwrap()).collect();

            let len_x = x_bounds.len() - 1;
            let len_y = y_bounds.len() - 1;
            let len_z = z_bounds.len() - 1;
            let len_energy = energy_mesh.len() - 1;

            return Some(Box::new(cartesian_mesh::CartesianMesh {
                x_bounds: x_bounds,
                y_bounds: y_bounds,
                z_bounds: z_bounds,
                energy_bounds: energy_mesh,
                tallies: Array4::<Tally>::zeros([len_x, len_y, len_z, len_energy]),
            }));
        } else if _mesh_type.eq("spherical") {
            let _center = _mesh_dict["center"].as_array().unwrap();
            let center: Vec<f64> = _center.iter().map(|f| f.as_f64().unwrap()).collect();

            let _r_bounds = _mesh_dict["r_bounds"].as_array().unwrap();
            let r_bounds: Vec<f64> = _r_bounds.iter().map(|f| f.as_f64().unwrap()).collect();

            let _theta_bounds = _mesh_dict["theta_bounds"].as_array().unwrap();
            let theta_bounds: Vec<f64> =
                _theta_bounds.iter().map(|f| f.as_f64().unwrap()).collect();

            let _phi_bounds = _mesh_dict["phi_bounds"].as_array().unwrap();
            let phi_bounds: Vec<f64> = _phi_bounds.iter().map(|f| f.as_f64().unwrap()).collect();

            let _energy_mesh = &self.dict["mesh"]["energy_mesh"].as_array().unwrap();
            let energy_mesh: Vec<f64> = _energy_mesh.iter().map(|f| f.as_f64().unwrap()).collect();

            let len_r = r_bounds.len() - 1;
            let len_theta = theta_bounds.len() - 1;
            let len_phi = phi_bounds.len() - 1;
            let len_energy = energy_mesh.len() - 1;

            return Some(Box::new(spherical_mesh::SphericalMesh {
                center: [center[0], center[1], center[2]],
                r_bounds: r_bounds,
                theta_bounds: theta_bounds,
                phi_bounds: phi_bounds,
                energy_bounds: energy_mesh,
                tallies: Array4::<Tally>::zeros([len_r, len_theta, len_phi, len_energy]),
            }));
        }
        None
    }
    pub fn get_variance_reduction_method(&self) -> String {
        let variance_reduction_dict = &self.dict["monte_carlo"]["variance_reduction"];
        variance_reduction_dict["method"].as_str().unwrap().to_string()
    }
    pub fn get_transport_mode(&self) -> String{
        let monte_carlo_dict = &self.dict["monte_carlo"];
        monte_carlo_dict["transport_mode"].as_str().unwrap().to_string()
    }
    pub fn get_ams_k_split(&self) -> usize {
        self.dict["monte_carlo"]["variance_reduction"]["ams_parameters"]["k_split"]
            .as_u64()
            .unwrap() as usize
    }
    pub fn get_strategy(&self) -> Option<strategy::Strategy> {
        if self.dict["monte_carlo"]["variance_reduction"]["strategy"] == serde_json::Value::Null {
            return None;
        }
        let startegy_input_ = &self.dict["monte_carlo"]["variance_reduction"]["strategy"];
        let scheme_: String = startegy_input_["scheme"].as_str().unwrap().to_string();

        #[allow(unused_assignments)]
        let mut startegy_scheme = strategy::Scheme::Bootstrap;

        // Retrieve reinforcement learning scheme
        if scheme_.eq("alternate") {
            startegy_scheme = strategy::Scheme::Alternate;
        } else if scheme_.eq("bootstrap") {
            startegy_scheme = strategy::Scheme::Bootstrap;
        } else {
            panic!("Define a scheme for the reinforcement learning part to run")
        }
        let mut phases = Vec::new();
        let mut lengths = HashMap::<strategy::Phase, u32>::new();

        if startegy_input_["phases"]["exploration"] != serde_json::Value::Null {
            phases.push(strategy::Phase::Explore);
            lengths.insert(
                strategy::Phase::Explore,
                startegy_input_["phases"]["exploration"]["length"]
                    .as_u64()
                    .unwrap() as u32,
            );
        }
        if startegy_input_["phases"]["exploitation"] != serde_json::Value::Null {
            phases.push(strategy::Phase::Exploit);
            lengths.insert(
                strategy::Phase::Exploit,
                startegy_input_["phases"]["exploitation"]["length"]
                    .as_u64()
                    .unwrap() as u32,
            );
        }

        let _startegy = strategy::Strategy {
            scheme: startegy_scheme,
            current_phase: strategy::Phase::Explore,
            phases: phases,
            phase_length: lengths,
            batch_index_at_last_update: 0,
        };
        Some(_startegy)
    }
    pub fn get_homogenizer(&self) -> Box<dyn homogenizer::Homogenizer> {
        if self.dict["monte_carlo"]["homogenizer"] == serde_json::Value::Null {
            return Box::new(homogenizer::Null {});
        }
        let homogenizer_input = &self.dict["monte_carlo"]["homogenizer"];
        let geom_disc_type: String = homogenizer_input["geometrical_discretization"]["type"]
            .as_str()
            .unwrap()
            .to_string();

        let library = CrossSectionLibrary::new(Some("multigroup".to_string()));
        let eb = homogenizer_input["energy_discretization"]
            .as_array()
            .unwrap();
        let energy_bounds: Vec<f64> = eb.iter().map(|f| f.as_f64().unwrap()).collect();

        let mut to_file = None;
        if homogenizer_input["to_file"] != serde_json::Value::Null {
            to_file = Some(homogenizer_input["to_file"].as_str().unwrap().to_string());
        }

        if geom_disc_type.eq("volume") {
            return Box::new(homogenizer::VolumeHomogenizer {
                cross_section_library: library,
                energy_bounds: energy_bounds,
                flux_tallies: HashMap::new(),
                total_xs_tallies: HashMap::new(),
                scattering_xs_tallies: HashMap::new(),
                scattering_matrix_tallies: HashMap::new(),
                to_file: to_file,
            });
        } else if geom_disc_type.eq("mesh") {
            return Box::new(homogenizer::Null {});
        }
        return Box::new(homogenizer::Null {});
    }

    pub fn get_cross_section_library(&self) -> CrossSectionLibrary {
        if self.dict["cross_section_library"] == serde_json::Value::Null {
            return CrossSectionLibrary {
                evaluation_type: EvaluationType::Punctual,
                isotopes: HashMap::new(),
            };
        }
        let xs_dict = &self.dict["cross_section_library"];
        if xs_dict["from_file"] != serde_json::Value::Null {
            let mut csl = CrossSectionLibrary::new(Some("multigroup".to_string()));
            csl.read_from_file(
                xs_dict["from_file"].as_str().unwrap().to_string(),
            );
            return csl;
        }
        panic!("no cross section library found");
    }
}
