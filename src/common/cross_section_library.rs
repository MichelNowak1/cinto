use serde_json::json;
use std::fs::File;
use std::io::Error;
use std::io::Write;
use colored::*;

use crate::common::cross_section::CrossSection;
use crate::common::cross_section::MultiGroupCrossSection;
use crate::common::cross_section::PunctualCrossSection;
use crate::common::geometry::Geometry;
use crate::common::isotope::Isotope;

use ndarray::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};
use std::collections::HashMap;
use std::sync::Arc;

/// Interaction type
#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub enum Interaction {
    Split,
    Source,
    Null,
    Fission,
    ElasticScattering,
    InelasticScattering,
    Scattering, // Elastic + Inelastic
    Absorption,
    Total,
    Reflection,
    Jump,
    Cross,
    Leak,
}

/// Library type
///
/// * Punctual
///     1) cross sections evaluated ponctually ordered with increasing values in energy
///     2) energy vector has same size of cross section vector
///     3) linear interpolation (trapezeoid integration rule) will be used for cross section
///    evaluation
///
/// * Multigroup
///     1) cross sections evaluated per group ordered with decreasing values in energy
///     2) energy vector has size of cross section vector + 1 (defining the bounds of the groups)
///     3) constant values per group (step integration rule)
#[derive(PartialEq, Clone, Copy)]
pub enum EvaluationType {
    Punctual,
    Multigroup,
}

/// Cross section library
///
/// # Arguments:
///
/// * `isotopes` map of mcnp_id of isotope to isotope;
/// * `evaluation_type` type of library (Punctual or Multigroup)
///
#[pyclass]
pub struct CrossSectionLibrary {
    pub isotopes: HashMap<String, Arc<Isotope>>,
    pub evaluation_type: EvaluationType,
}

#[pymethods]
impl CrossSectionLibrary {
    /// Create new cross section library
    ///
    /// # Arguments:
    ///
    /// * `evaluation_type` type of the library (Punctual or Multigroup)
    ///
    /// # Examples
    /// ```
    /// use cinto::common::cross_section_library::CrossSectionLibrary;
    ///
    /// let cross_section_library = CrossSectionLibrary::new("multigroup".to_string());
    /// ```
    #[new]
    pub fn new(evaluation_type_str: Option<String>) -> Self {
        let evaluation_type = if evaluation_type_str.is_some() {
            match evaluation_type_str.unwrap().as_str(){
                "multigroup" => EvaluationType::Multigroup,
                "punctual"  => EvaluationType::Punctual,
                _ => panic!("Evaluation type not recognized"),
            }
        } else {
            EvaluationType::Punctual
        };

        CrossSectionLibrary {
            isotopes: HashMap::<String, Arc<Isotope>>::new(),
            evaluation_type: evaluation_type,
        }

        // println!("initializing macroscopic cross sections");
        // for material in &mut self.materials {
        //     material.init_macroscopic_cross_sections(self.cross_section_library_type);
        // }
    }
    /// Adds an isotope to the cross section library
    ///
    /// this methods wraps the isotopes in the cinto_pyne library
    /// the list of isotopes that can be added can be found in:
    /// cinto/docs/list_of_materials.txt
    ///
    /// # Arguments:
    ///
    /// * `name` name of the isotope to be added
    ///
    /// # Examples
    /// ```
    /// use cinto::common::cross_section_library::CrossSectionLibrary;
    ///
    /// let mut cross_section_library = CrossSectionLibrary::new("punctual");
    ///
    /// cross_section_library.add_isotope("Water, Liquid")
    /// ```
    pub fn add_isotope(&mut self, name: String) -> bool {
        // if isotope is already in library, return
        if self.isotopes.get(&name).is_some() {
            return true;
        }

        Python::with_gil(|py| {
            let cinto_pyne = py.import("cinto").unwrap();

            let py_isotope_name = PyString::new(py, &name);

            let mut evaluation_type = PyString::new(py, "Punctual");
            if self.evaluation_type == EvaluationType::Multigroup {
                evaluation_type = PyString::new(py, "Multigroup");
            }

            let py_isotope = cinto_pyne
                .getattr("get_isotope")
                .unwrap()
                .call1((py_isotope_name, evaluation_type))
                .unwrap()
                .extract::<PyObject>()
                .unwrap();

            let found: bool = py_isotope
                .getattr(py, "is_some")
                .unwrap()
                .extract(py)
                .unwrap();

            if !found {
                self.isotopes.remove(&name);
                return false;
            }

            let atomic_mass = py_isotope
                .getattr(py, "atomic_mass")
                .unwrap()
                .extract(py)
                .unwrap();

            let energies: Vec<f64> = py_isotope
                .getattr(py, "energies")
                .unwrap()
                .extract(py)
                .unwrap();

            let interactions: Vec<String> = py_isotope
                .getattr(py, "interactions")
                .unwrap()
                .extract(py)
                .unwrap();

            let is_fissile: bool = py_isotope
                .getattr(py, "is_fissile")
                .unwrap()
                .extract(py)
                .unwrap();

            let mut isotope = Isotope {
                name: name.clone(),
                atomic_mass: atomic_mass,
                energies: energies,
                is_fissile: is_fissile,
                cross_sections: HashMap::new(),
                scattering_matrix: None,
            };

            for i in 0..interactions.len() {
                let interaction = match interactions[i].as_str() {
                    "Total" => Interaction::Total,
                    "ElasticScattering" => Interaction::ElasticScattering,
                    _ => panic!("interaction: {} not implemented yet", interactions[i]),
                };

                let cross_section_values: Vec<f64> = py_isotope
                    .call_method1(py, "get_cross_sections", PyTuple::new(py, &[i]))
                    .unwrap()
                    .extract(py)
                    .unwrap();

                match self.evaluation_type {
                    EvaluationType::Punctual => {
                        let cross_section = PunctualCrossSection {
                            values: cross_section_values,
                        };
                        isotope
                            .cross_sections
                            .insert(interaction, Box::new(cross_section));
                    }

                    EvaluationType::Multigroup => {
                        let cross_section = MultiGroupCrossSection {
                            values: cross_section_values,
                        };
                        isotope
                            .cross_sections
                            .insert(interaction, Box::new(cross_section));
                    }
                }
            }

            let total_xs = isotope.cross_sections.get(&Interaction::Total).unwrap();
            let mut abs_xs = total_xs.get_values().clone();

            for (interaction, xs) in &isotope.cross_sections {
                if interaction == &Interaction::Total {
                    continue;
                }
                abs_xs = abs_xs
                    .iter()
                    .zip(xs.get_values().iter())
                    .map(|(&abs, &xs)| abs - xs)
                    .collect();
            }
            match self.evaluation_type {
                EvaluationType::Punctual => {
                    let cross_section = PunctualCrossSection { values: abs_xs };
                    isotope
                        .cross_sections
                        .insert(Interaction::Absorption, Box::new(cross_section));
                }

                EvaluationType::Multigroup => {
                    let cross_section = MultiGroupCrossSection { values: abs_xs };
                    isotope
                        .cross_sections
                        .insert(Interaction::Absorption, Box::new(cross_section));
                }
            }

            self.isotopes.insert(name, Arc::new(isotope));
            return true;
        })
    }

    pub fn init(&mut self, geometry: &mut Geometry){
        // fill the cross section library with needed isotopes if not built yet
        for material in &mut geometry.materials {
            let mut not_found:Vec<usize> = Vec::new();
            for (i, isotope_name) in material.isotope_names.iter().enumerate() {
                let found = self.add_isotope(isotope_name.to_string());
                if !found {
                    println!(
                        " {} {} {}",
                        "Warning: isotope".bold().yellow(),
                        isotope_name,
                        "not found, will be bypassed for calculation"
                        .bold()
                        .yellow()
                        );
                    not_found.push(i);
                }
            }

            // remove isotopes not found in library
            for i in (0..not_found.len()).rev() {
                material.isotope_names.remove(not_found[i]);
                material.isotope_concentrations.remove(not_found[i]);
            }
        }
    }
    /// Reads cross section library from file.
    ///
    /// # Arguments:
    ///
    /// * `file_name` name of the json formated file containing the library.
    ///
    /// # Examples
    /// ```
    /// use cinto::common::cross_section_library::CrossSectionLibrary;
    ///
    /// let cross_section_library = CrossSectionLibrary::new("multigroup".to_string());
    /// let xs_lib = cross_section_library.read_from_file("xs_lib.json");
    /// ```
    pub fn read_from_file(&mut self, file_name: String) {
        let file = File::open(file_name).expect("file should open read only");
        let dict: serde_json::Value =
            serde_json::from_reader(file).expect("file should be proper JSON");

        // retrieve evaluation type
        let evaluation_type_string = &dict["evaluation_type"];
        let mut evaluation_type = EvaluationType::Multigroup;
        if evaluation_type_string.eq("punctual") {
            evaluation_type = EvaluationType::Punctual;
        } else if evaluation_type_string.eq("multigroup") {
            evaluation_type = EvaluationType::Multigroup;
        }

        // read isotopes
        let mut isotopes: HashMap<String, Arc<Isotope>> = HashMap::new();

        for isotope_reader in dict["isotopes"].as_array().unwrap() {
            let mut cross_sections: HashMap<Interaction, Box<dyn CrossSection + Send + Sync>> =
                HashMap::new();
            let name: String = isotope_reader["name"]
                .as_str()
                .unwrap()
                .to_string();

            let atomic_mass = isotope_reader["atomic_mass"].as_f64().unwrap();
            let is_fissile = isotope_reader["is_fissile"].as_bool().unwrap();

            let energies_read = &isotope_reader["energies"].as_array().unwrap();
            let energies: Vec<f64> = energies_read.iter().map(|f| f.as_f64().unwrap()).collect();

            let total_xs_read = &isotope_reader["total_xs"].as_array().unwrap();
            let total_xs_values: Vec<f64> =
                total_xs_read.iter().map(|f| f.as_f64().unwrap()).collect();

            let scattering_xs_read = &isotope_reader["scattering_xs"].as_array().unwrap();
            let scattering_xs_values: Vec<f64> =
                scattering_xs_read.iter().map(|f| f.as_f64().unwrap()).collect();

            let fission_xs_read = &isotope_reader["fission_xs"].as_array().unwrap();
            let fission_xs_values: Vec<f64> = fission_xs_read
                .iter()
                .map(|f| f.as_f64().unwrap())
                .collect();

            // scattering
            let num_energy_groups = energies.len() - 1;

            let scattering_matrix_read = &isotope_reader["scattering_matrix"];
            let mut scattering_matrix =
                Array2::<f64>::zeros((num_energy_groups, num_energy_groups));

            for (i, row) in scattering_matrix_read
                .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    {
                        for (j, element) in row.as_array().unwrap().iter().enumerate() {
                            scattering_matrix[(i, j)] = element.as_f64().unwrap();
                        }
                    }
            let scattering_xs_values = scattering_matrix.sum_axis(Axis(1)).to_vec();

            let mut abs_xs_values = Vec::new();
            for i in 0..total_xs_values.len() {
                abs_xs_values
                    .push(total_xs_values[i] - fission_xs_values[i] - scattering_xs_values[i]);
            }
            if evaluation_type == EvaluationType::Multigroup {
                cross_sections.insert(
                    Interaction::Total,
                    Box::new(MultiGroupCrossSection {
                        values: total_xs_values,
                    }),
                    );
            } else if evaluation_type == EvaluationType::Punctual {
                cross_sections.insert(
                    Interaction::Total,
                    Box::new(PunctualCrossSection {
                        values: total_xs_values,
                    }),
                    );
            }
            if evaluation_type == EvaluationType::Multigroup {
                cross_sections.insert(
                    Interaction::Fission,
                    Box::new(MultiGroupCrossSection {
                        values: fission_xs_values,
                    }),
                    );
            } else if evaluation_type == EvaluationType::Punctual {
                cross_sections.insert(
                    Interaction::Fission,
                    Box::new(PunctualCrossSection {
                        values: fission_xs_values,
                    }),
                    );
            }
            if evaluation_type == EvaluationType::Multigroup {
                cross_sections.insert(
                    Interaction::ElasticScattering,
                    Box::new(MultiGroupCrossSection {
                        values: scattering_xs_values,
                    }),
                    );
            } else if evaluation_type == EvaluationType::Punctual {
                cross_sections.insert(
                    Interaction::ElasticScattering,
                    Box::new(PunctualCrossSection {
                        values: scattering_xs_values,
                    }),
                    );
            }
            if evaluation_type == EvaluationType::Multigroup {
                cross_sections.insert(
                    Interaction::Absorption,
                    Box::new(MultiGroupCrossSection {
                        values: abs_xs_values,
                    }),
                    );
            } else if evaluation_type == EvaluationType::Punctual {
                cross_sections.insert(
                    Interaction::Absorption,
                    Box::new(PunctualCrossSection {
                        values: abs_xs_values,
                    }),
                    );
            }

            let isotope = Isotope {
                name: name.to_string(),
                atomic_mass: atomic_mass,
                energies: energies,
                is_fissile: is_fissile,
                cross_sections: cross_sections,
                scattering_matrix: Some(scattering_matrix),
            };
            isotopes.insert(name, Arc::new(isotope));
        }

        self.evaluation_type = evaluation_type;
        self.isotopes = isotopes;
    }
}
impl CrossSectionLibrary {

    /// Returns an isotope from its name.
    ///
    /// # Arguments:
    ///
    /// * `isotope_name` isotope name
    ///
    /// # Examples
    /// ```
    /// use cinto::common::cross_section_library::CrossSectionLibrary;
    ///
    /// let cross_section_library = CrossSectionLibrary::new("multigroup".to_string());
    ///
    /// let isotope = cross_section_library.get_isotope("dummy");
    /// ```
    pub fn get_isotope(&self, isotope_name: &String) -> Arc<Isotope> {
        let isotope = self.isotopes.get(isotope_name);
        if isotope.is_some() {
            isotope.unwrap().clone()
        } else {
            panic!(
                "isotope looked for not present in cross section library: {}",
                isotope_name
            );
        }
    }
    /// Saved the cross section library to a json formated file.
    ///
    /// # Arguments:
    ///
    /// * `isotope_name` isotope name
    ///
    /// # Examples
    /// ```
    /// use cinto::common::cross_section_library::CrossSectionLibrary;
    ///
    /// let cross_section_library = CrossSectionLibrary::new("multigroup".to_string());
    ///
    /// let isotope = cross_section_library.get_isotope("dummy");
    /// ```
    pub fn save_to_file(&self, file_name: String) -> Result<(), Error> {
        let mut output_file = File::create(file_name)?;
        let mut eval_string = "punctual";

        if self.evaluation_type == EvaluationType::Multigroup {
            eval_string = "multigroup";
        }
        let mut isotopes_json = Vec::new();
        for (isotope_name, isotope) in &self.isotopes {
            // convert scattering matrix to Vec<Vec<f64>> for serialization
            let mut scattering_matrix: Vec<Vec<f64>> = Vec::new();
            for row_index in 0..isotope.scattering_matrix.as_ref().unwrap().nrows() {
                scattering_matrix.push(
                    isotope
                        .scattering_matrix
                        .as_ref()
                        .unwrap()
                        .row(row_index)
                        .to_vec(),
                );
            }
            let isotope_json = json!({
                "name": isotope_name,
                "energies": isotope.energies,
                "total_xs": isotope.cross_sections.get(&Interaction::Total)
                                                  .unwrap()
                                                  .get_values(),
                "scattering_xs": isotope.cross_sections.get(&Interaction::ElasticScattering)
                                                       .unwrap()
                                                       .get_values(),
                "fission_xs": isotope.cross_sections.get(&Interaction::Fission)
                                                    .unwrap()
                                                    .get_values(),
                "scattering_matrix": scattering_matrix,
            });
            isotopes_json.push(isotope_json.clone());
        }
        let xs_library_json = json!({
            "evaluation_type": eval_string,
            "isotopes": isotopes_json,
        });
        let string = format!(
            "{}",
            serde_json::to_string_pretty(&xs_library_json).unwrap()
        );
        write!(output_file, "{}\n", string).unwrap();
        Ok(())
    }
}
