use itertools::iproduct;
use ndarray::*;
use ndarray_linalg::*;
use std::fs::File;
use std::io::Write;

use crate::common::calculation_type::CalculationType;
use crate::common::cartesian_mesh::CartesianMesh;
use crate::common::cross_section_library::CrossSectionLibrary;
use crate::common::cross_section_library::Interaction;
use crate::common::detector::Detector;
use crate::common::geometry::Geometry;
use crate::common::geometry::BoundaryCondition;
use crate::common::mesh::Mesh;
use crate::common::profiler;
use crate::common::source::Source;
use crate::monte_carlo::trajectory::Point;

use colored::*;
use console::style;

use serde_json::json;
use std::io::Error;

#[derive(PartialEq)]
pub enum CalculationMode{
    Direct,
    Adjoint,
}
pub enum Boundary {
    XMin,
    XMax,
    YMin,
    YMax,
    ZMin,
    ZMax,
}

pub struct FiniteDifferencesSolver {
    pub calculation_mode: CalculationMode,
    pub calculation_type: CalculationType,
    pub iteration: usize,
    pub coefficients: Array2<f64>,
    pub mesh: CartesianMesh,

    pub num_regions: usize,
    pub num_groups: usize,
}

impl FiniteDifferencesSolver {
    pub fn new(
        calculation_mode: CalculationMode,
        calculation_type: CalculationType,
        spatial_mesh: CartesianMesh,
    ) -> FiniteDifferencesSolver {
        let mut num_regions: usize = spatial_mesh.x_bounds.len() - 1;
        num_regions *= spatial_mesh.y_bounds.len() - 1;
        num_regions *= spatial_mesh.z_bounds.len() - 1;
        let num_groups: usize = spatial_mesh.energy_bounds.len() - 1;

        FiniteDifferencesSolver {
            calculation_mode: calculation_mode,
            calculation_type: calculation_type,
            iteration: 0,
            coefficients: Array2::<f64>::zeros((10, 10)),
            mesh: spatial_mesh,
            num_regions: num_regions,
            num_groups: num_groups,
        }
    }

    pub fn compute_coefficients(
        &mut self,
        dim: [usize; 3],
        total_cross_sections: Array1<f64>,
        removal_cross_sections: Array1<f64>,
        diffusion_coefficients: Array1<f64>,
        boundary_conditions: [[BoundaryCondition; 2]; 3],
    ) -> Array2<f64> {
        let mut num_regions = 1;
        let mut deltas: Vec<Vec<f64>> = Vec::new();

        let dimension: usize = dim.iter().sum();

        num_regions *= self.mesh.x_bounds.len() - 1;
        let mut axis_deltas = Vec::new();
        for i in 0..self.mesh.x_bounds.len() - 1 {
            axis_deltas.push(self.mesh.x_bounds[i + 1] - self.mesh.x_bounds[i]);
        }
        deltas.push(axis_deltas);

        num_regions *= self.mesh.y_bounds.len() - 1;
        axis_deltas = Vec::new();
        for i in 0..self.mesh.y_bounds.len() - 1 {
            axis_deltas.push(self.mesh.y_bounds[i + 1] - self.mesh.y_bounds[i]);
        }
        deltas.push(axis_deltas);

        num_regions *= self.mesh.z_bounds.len() - 1;
        axis_deltas = Vec::new();
        for i in 0..self.mesh.z_bounds.len() - 1 {
            axis_deltas.push(self.mesh.z_bounds[i + 1] - self.mesh.z_bounds[i]);
        }
        deltas.push(axis_deltas);

        let mut solving_matrix = Array2::<f64>::zeros((num_regions, num_regions));

        let nx = deltas[0].len();
        let ny = deltas[1].len();
        let nz = deltas[2].len();
        let dimensions = vec![nx, ny, nz];

        for (ix, iy, iz) in iproduct!(0..nx, 0..ny, 0..nz) {
            let region = ix + nx * iy + nx * ny * iz;
            let coordinates = vec![ix, iy, iz];

            let (delta_inner, delta_lower, delta_upper) =
                self.get_local_deltas(coordinates, &dimensions, &deltas);

            let (d_inner, d_lower, d_upper, r_lower, r_upper) = self
                .get_local_diffusion_coefficients(
                    ix,
                    iy,
                    iz,
                    nx,
                    ny,
                    nz,
                    &diffusion_coefficients,
                );

            solving_matrix[(region, region)] = removal_cross_sections[region];

            for d in 0..3 {

                if region != r_lower[d]{

                    let lower_coeff = 2. * d_lower[d] * d_inner
                        / (delta_inner[d] * d_lower[d] + delta_lower[d] * d_inner);

                    solving_matrix[(region, r_lower[d])] = - lower_coeff;
                    solving_matrix[(region, region)] += lower_coeff;
                }

                if region != r_upper[d] {

                    let upper_coeff = 2. * d_upper[d] * d_inner
                        / (delta_inner[d] * d_upper[d] + delta_upper[d] * d_inner);

                    solving_matrix[(region, r_upper[d])] = - upper_coeff;
                    solving_matrix[(region, region)] += upper_coeff;
                }

                let boundary = match d {
                    0 => if ix == 0 {0} else if ix == nx - 1 {1} else {-1_i32} ,
                    1 => if iy == 0 {0} else if iy == ny - 1 {1} else {-1_i32} ,
                    2 => if iz == 0 {0} else if iz == nz - 1 {1} else {-1_i32} ,
                    _ => panic!("unrecognized dimension {}", d),
                };

                if boundary >= 0 {
                    if boundary_conditions[d][boundary as usize] == BoundaryCondition::Leak{
                        solving_matrix[(region, region)] += 2. * d_inner / delta_inner[d];
                    }
                }
            }
        }
        solving_matrix
    }

    pub fn solve(
        &mut self,
        output_file: &mut File,
        geometry: &Geometry,
        cross_section_library: &CrossSectionLibrary,
        sources: &Vec<Source>,
        detectors: &mut Vec<Detector>,
        meshes: &mut Vec<Box<dyn Mesh>>,
    ) {
        let nx = self.mesh.x_bounds.len() - 1;
        let ny = self.mesh.y_bounds.len() - 1;
        let nz = self.mesh.z_bounds.len() - 1;
        let num_regions = nx * ny * nz;
        let num_groups = self.mesh.energy_bounds.len() - 1;

        // compute problem dimension
        let mut dim: [usize; 3] = [1, 1, 1];

        if nx == 1 {
            dim[0] = 0;
        }
        if ny == 1 {
            dim[1] = 0;
        }
        if nz == 1 {
            dim[2] = 0;
        }

        let dimension = dim[0] + dim[1] + dim[2];

        let mut total_cross_sections = Array2::<f64>::zeros((num_regions, num_groups));
        let mut scattering_cross_sections = Array2::<f64>::zeros((num_regions, num_groups));
        let mut self_scattering_xs = Array2::<f64>::zeros((num_regions, num_groups));
        let mut scattering_matrices = Array3::<f64>::zeros((num_regions, num_groups, num_groups));

        // iterate over all cells (geometric and energetic) to
        // fill cross section tables
        println!(
            "{} {} {}",
            style("->").bold().dim(),
            "\u{26F3}",
            "Generating cross section tables...".bold().blue()
        );
        for (ix, iy, iz, g) in iproduct!(0..nx, 0..ny, 0..nz, 0..self.num_groups) {
            // local index
            let r = ix + nx * iy + nx * ny * iz;

            let middle_x = (self.mesh.x_bounds[ix + 1] + self.mesh.x_bounds[ix]) / 2.;
            let middle_y = (self.mesh.y_bounds[iy + 1] + self.mesh.y_bounds[iy]) / 2.;
            let middle_z = (self.mesh.z_bounds[iz + 1] + self.mesh.z_bounds[iz]) / 2.;
            let middle_energy = (self.mesh.energy_bounds[g] + self.mesh.energy_bounds[g + 1]) / 2.;

            let material = geometry.get_material([middle_x, middle_y, middle_z]);

            let total_xs =
                material.get_sigma(Interaction::Total, middle_energy, &cross_section_library);

            let scattering_xs = material.get_sigma(
                Interaction::ElasticScattering,
                middle_energy,
                &cross_section_library,
            );

            let isotope = cross_section_library
                .isotopes
                .get(&material.isotope_names[0])
                .unwrap();

            let scat_m = isotope.scattering_matrix.as_ref().unwrap();
            for g1 in 0..scat_m.ncols() {
                for g2 in 0..scat_m.nrows() {
                    scattering_matrices[(r, g1, g2)] = scat_m[(g1, g2)];
                }
            }
            let full_scat = scattering_matrices.slice(s![r, .., ..]);
            let d = full_scat.diag();

            self_scattering_xs[(r, g)] = d[g];

            total_cross_sections[(r, g)] = total_xs;
            scattering_cross_sections[(r, g)] = scattering_xs;
        }

        let removal_cross_sections = total_cross_sections.clone() - self_scattering_xs;
        let diffusion_coefficients = 1. / (3. * total_cross_sections.clone());

        let mut flux = Array2::<f64>::zeros((num_regions, num_groups));

        let boundary_conditions = geometry.boundary_conditions;

        let num_outer_iterations = 1;
        // external iterations

        let mut progress_bar = profiler::CintoProgressBar::new(self.num_groups);
        
        let save_linear_sytem = false;

        // multi groups iterations
        for group in 0..self.num_groups {
            progress_bar.update(group as u64);

            let solving_matrix = self.compute_coefficients(
                dim,
                total_cross_sections.slice(s![.., group]).to_owned(),
                removal_cross_sections.slice(s![.., group]).to_owned(),
                diffusion_coefficients.slice(s![.., group]).to_owned(),
                boundary_conditions,
                );

            let mut target = Array1::<f64>::zeros(self.num_regions);

            self.add_external_source(&mut target, geometry, sources, detectors, group);

            self.add_scattering_source(
                &mut target,
                &flux,
                scattering_matrices.slice(s![.., group, ..]).to_owned(),
                group);

            if save_linear_sytem {
                let mut matrix_file = File::create("finite_differences_solving_matrix.txt").unwrap();
                for r in 0..self.num_regions {
                    for r in 0..self.num_regions {
                        writeln!(matrix_file, "{:?}", solving_matrix[(r,r)]).unwrap();
                    }
                }

                let mut target_file = File::create("finite_differences_target.txt").unwrap();
                for r in 0..self.num_regions {
                    writeln!(target_file, "{:?}", target[r]).unwrap();
                }
            }

            let result = self.solve_inner(solving_matrix, target.clone());

            for r in 0..self.num_regions {
                flux[(r, group)] = result[r];
            }
        }

        self.score_detectors(detectors);

        self.score_meshes(meshes);

        self.write_to_file(flux, "finite_differences.json".to_string());

        progress_bar.finish();
    }

    pub fn print_results(&self, flux: Array2<f64>) -> Result<(), Error> {
        let mut output_file = File::create("finite_differences.json")?;

        write!(output_file, "{{\n").unwrap();

        for group in 0..self.num_groups {
            // println!("group {}", group);
            println!("{:?}", flux.slice(s![.., group]).to_vec());
            let _detector_response = json!({
                format!("group {}", group): flux.slice(s![.., group]).to_vec()
            });
            let mut _response_string = format!(
                "{}",
                serde_json::to_string_pretty(&_detector_response).unwrap()
            );
            _response_string = _response_string.replace("{", "");
            _response_string = _response_string.replace("}", "");
            if group < self.num_groups - 1 {
                write!(output_file, "{},\n", _response_string).unwrap();
            } else {
                write!(output_file, "{}\n", _response_string).unwrap();
            }
        }

        write!(output_file, "}}\n").unwrap();

        Ok(())
    }

    fn write_to_file(&self,flux: Array2<f64>, file_name: String) -> Result<(), Error> {

        let mut output_file = File::create(file_name)?;

        // loop on the whole array of importances
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];
        let z: Vec<f64> = vec![];
        let e: Vec<f64> = vec![];
        let mut importance: Vec<f64> = vec![];
        for (i, j, k, l) in iproduct!(
            0..self.mesh.x_bounds.len() - 1,
            0..self.mesh.y_bounds.len() - 1,
            0..self.mesh.z_bounds.len() - 1,
            0..self.mesh.energy_bounds.len() -1
        ) {
            importance.push(flux.slice(s![.., l]).to_vec()[i+(self.mesh.x_bounds.len() -1)*j]);
        }

        let importance_map_json = json!({
            "type": "scalar",
            "x_bounds": self.mesh.x_bounds,
            "y_bounds": self.mesh.y_bounds,
            "z_bounds": self.mesh.z_bounds,
            "energy_bounds": self.mesh.energy_bounds,
            "importance": importance
        });
        let _response_string = format!(
            "{}",
            serde_json::to_string_pretty(&importance_map_json).unwrap()
        );
        write!(output_file, "{}", _response_string).unwrap();

        Ok(())
    }

    pub fn solve_inner(
        &mut self,
        solving_matrix: Array2<f64>,
        target: Array1<f64>,
    ) -> Array1<f64> {
        let solution = solving_matrix.solve_into(target.clone());

        solution.unwrap()
    }

    pub fn add_external_source(
        &mut self,
        target: &mut Array1<f64>,
        geometry: &Geometry,
        sources: &Vec<Source>,
        detectors: &Vec<Detector>,
        current_energy_group: usize,
    ) {
        match self.calculation_mode {
            CalculationMode::Direct => {
                 let particle = sources[0].sample();
                 let ip= Point{
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
                 };
                 let index = self.mesh.get_index(&ip).unwrap();

                 if current_energy_group != index[3] {
                     return;
                 }

                 let nx = self.mesh.x_bounds.len() - 1;
                 let ny = self.mesh.y_bounds.len() - 1;
                 let nz = self.mesh.z_bounds.len() - 1;

                 let r = index[0] + nx * index[1] + nx * ny * index[2];
                 target[r] += 1.;
            }
            CalculationMode::Adjoint => {
                let nx = self.mesh.x_bounds.len() - 1;
                let ny = self.mesh.y_bounds.len() - 1;
                let nz = self.mesh.z_bounds.len() - 1;

                for (ix, iy, iz) in iproduct!(0..nx, 0..ny, 0..nz) {
                    let r = ix + nx * iy + nx * ny * iz;

                    let middle_x = (self.mesh.x_bounds[ix + 1] + self.mesh.x_bounds[ix]) / 2.;
                    let middle_y = (self.mesh.y_bounds[iy + 1] + self.mesh.y_bounds[iy]) / 2.;
                    let middle_z = (self.mesh.z_bounds[iz + 1] + self.mesh.z_bounds[iz]) / 2.;

                    let volume_name = geometry.get_volume_name([middle_x, middle_y, middle_z]);

                    for detector in detectors {
                        if volume_name.eq(&detector.volume_name) {
                            let upper_energy = detector.energy_bounds[0];
                            let lower_energy = detector.energy_bounds.last().unwrap();

                            if self.mesh.energy_bounds[current_energy_group] <= upper_energy
                                && self.mesh.energy_bounds[current_energy_group] >= *lower_energy
                            {
                                target[r] += 1.;
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn add_scattering_source(
        &self,
        target: &mut Array1<f64>,
        flux: &Array2<f64>,
        scattering_cross_sections: Array2<f64>,
        group: usize,
    ) {
        for g in 0..self.num_groups {
            // do not add self scattering to the target
            if g == group {
                continue;
            }
            for r in 0..self.num_regions {
                target[r] += flux[(r, g)] * scattering_cross_sections[(r, g)];
            }
        }
    }

    pub fn get_local_deltas(
        &self,
        coordinates: Vec<usize>,
        dimensions: &Vec<usize>,
        deltas: &Vec<Vec<f64>>,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut delta_inner = Vec::new();
        let mut delta_lower = Vec::new();
        let mut delta_upper = Vec::new();

        let dim = coordinates.len();

        for d in 0..dim {
            let center = coordinates[d];
            delta_inner.push(deltas[d][center]);

            if center > 0 {
                delta_lower.push(deltas[d][center - 1]);
            } else {
                delta_lower.push(deltas[d][center]);
            }

            if center < dimensions[d] - 1 {
                delta_upper.push(deltas[d][center + 1]);
            } else {
                delta_upper.push(deltas[d][center]);
            }
        }
        (delta_inner, delta_lower, delta_upper)
    }

    pub fn get_local_diffusion_coefficients(
        &self,
        ix: usize,
        iy: usize,
        iz: usize,
        nx: usize,
        ny: usize,
        nz: usize,
        d: &Array1<f64>,
    ) -> (f64, Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>) {
        let r = ix + nx * iy + nx * ny * iz;

        let rx_lower = {
            if ix > 0 {
                ix - 1 + nx * iy + nx * ny * iz
            } else {
                r
            }
        };
        let rx_upper = {
            if ix < nx - 1 {
                ix + 1 + nx * iy + nx * ny * iz
            } else {
                r
            }
        };
        let ry_lower = {
            if iy > 0 {
                ix + nx * (iy - 1) + nx * ny * iz
            } else {
                r
            }
        };
        let ry_upper = {
            if iy < ny - 1 {
                ix + nx * (iy + 1) + nx * ny * iz
            } else {
                r
            }
        };
        let rz_lower = {
            if iz > 0 {
                ix + nx * iy + nx * ny * (iz - 1)
            } else {
                r
            }
        };
        let rz_upper = {
            if iz < nz - 1 {
                ix + nx * iy + nx * ny * (iz + 1)
            } else {
                r
            }
        };

        let d_inner = d[r];
        let d_lower = vec![d[rx_lower], d[ry_lower], d[rz_lower]];
        let d_upper = vec![d[rx_upper], d[ry_upper], d[rz_upper]];
        let r_lower = vec![rx_lower, ry_lower, rz_lower];
        let r_upper = vec![rx_upper, ry_upper, rz_upper];

        (d_inner, d_lower, d_upper, r_lower, r_upper)
    }

    /// Computes lower diagonal coefficients of solving matrix on one axis
    ///
    /// # Arguments
    /// * `d_inner` inner diffusion coefficient
    /// * `d_lower` diffusion coefficient of lower cell
    /// * `delta_inner` inner cell size
    /// * `delta_lower` lower cell size on given axis
    ///
    /// # Returns
    /// * `coefficient` returns the coefficient to fill the solving matrix
    pub fn compute_neighbour_coefficient(
        &self,
        d_inner: f64,
        d_neighbour: f64,
        delta_inner: f64,
        delta_neighbour: f64,
    ) -> f64 {
        2. * d_neighbour * d_inner
            / (delta_inner * d_neighbour + delta_neighbour * d_inner)
            / delta_inner
    }

    pub fn score_detectors(&mut self, detectors: &mut Vec<Detector>) {
        // TODO
    }

    pub fn score_meshes(&mut self, meshes: &mut Vec<Box<dyn Mesh>>) {
        // TODO
    }
}
