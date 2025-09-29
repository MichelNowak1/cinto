extern crate libc;

use ndarray::prelude::*;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate pyo3;

#[macro_use]
extern crate serde_json;
#[macro_use]
extern crate itertools;

use std::env;

use cinto::common;
use cinto::input;
use cinto::deterministic::finite_differences::FiniteDifferencesSolver;
use cinto::deterministic::finite_differences::CalculationMode;
use cinto::common::calculation_type::CalculationType;
use cinto::monte_carlo::solver::MonteCarloSolver;
use cinto::monte_carlo::tally::Tally;


use colored::*;
use console::style;
use std::fs::File;
use std::io::Error;
use std::io::Write;

fn main() -> Result<(), Error> {
    println!(
        "{} {} {}",
        style("->").bold().dim(),
        "\u{2699}",
        "Initializing general parameters...".bold().blue()
    );

    let args: Vec<_> = env::args().collect();
    if args.len() != 2 {
        panic!("Provide an input file for the code to run");
    }
    let input = input::json_input::JSONInput::new(args[1].to_string());

    let _name = input.get_name();
    let name = json!({ "name": _name });

    let _output_file_name = input.get_output_file_name();

    let mut output_file = File::create(_output_file_name)?;

    let mut string = format!("{}", name);
    string = string.replace("{", "");
    string = string.replace("}", "");
    write!(&mut output_file, "{{\n")?;
    write!(&mut output_file, "{},\n", string)?;
    write!(&mut output_file, "\"result\":[\n")?;

    // Create geometry
    println!(
        "{} {} {}",
        style("->").bold().dim(),
        "\u{26EB}",
        "Initializing geometry and loading cross sections..."
            .bold()
            .blue()
    );
    let mut cross_section_library = input.get_cross_section_library();
    let mut geometry = input.get_geometry(&mut cross_section_library);
    geometry.init();
    cross_section_library.init(&mut geometry);

    // get source
    println!(
        "{} {} {}",
        style("->").bold().dim(),
        "\u{26A1}",
        "Initializing sources...".bold().blue()
    );
    let source = input.get_source().unwrap();
    let sources = vec![source];

    // Load detectors
    let mut detectors = Vec::new();
    let _detector = input.get_detector();
    if _detector.is_some() {
        detectors.push(_detector.unwrap());
    }

    // Load meshes
    let mut meshes = Vec::new();
    let _mesh = input.get_mesh();
    if _mesh.is_some() {
        meshes.push(_mesh.unwrap());
    }

    // Monte Carlo
    let mut monte_carlo_solver: Option<MonteCarloSolver> = None;
    if input.dict["monte_carlo"] != serde_json::Value::Null {
        monte_carlo_solver = Some(MonteCarloSolver::new(
            input.get_calculation_type(),
            input.get_num_batches(),
            input.get_num_particles_per_batch(),
            input.get_variance_reduction_method(),
            input.get_transport_mode()
        ))
    }

    // Create importance map
    println!(
        "{} {} {}",
        style("->").bold().dim(),
        "\u{1F511}",
        "Initializing importance maps...".bold().blue()
    );
    let importance_map = input.get_importance("importance".to_string());

    let scored_importance = input.get_importance("scored_importance".to_string());

    let mut strategy = input.get_strategy();

    let mut homogenizer = input.get_homogenizer();
    if homogenizer.is_some() {
        homogenizer.init(&geometry);
    }

    // run solvers
    if monte_carlo_solver.is_some() {
        monte_carlo_solver.unwrap().solve(
            &mut output_file,
            &geometry,
            &cross_section_library,
            &sources,
            &mut detectors,
            &mut meshes,
            importance_map,
            scored_importance,
            &mut strategy,
            &mut homogenizer,
        );
    };

    // Deterministic
    let deterministicinput = &input.dict["deterministic"];
    let mut deterministic_solver = None;
    if deterministicinput != &serde_json::Value::Null {
        let _solver_type = &deterministicinput["solver_type"];

        let calculation_type = input.get_calculation_type();

        let ct = CalculationType::Shielding;

        let mut calculation_mode = CalculationMode::Direct;
        if deterministicinput["calculation_mode"].eq("adjoint") {
            calculation_mode = CalculationMode::Adjoint;
        }
        let _mesh_dict = &deterministicinput["spatial_mesh"];

        let _x_bounds = _mesh_dict["x_bounds"].as_array().unwrap();
        let x_bounds: Vec<f64> = _x_bounds.iter().map(|f| f.as_f64().unwrap()).collect();

        let _y_bounds = _mesh_dict["y_bounds"].as_array().unwrap();
        let y_bounds: Vec<f64> = _y_bounds.iter().map(|f| f.as_f64().unwrap()).collect();

        let _z_bounds = _mesh_dict["z_bounds"].as_array().unwrap();
        let z_bounds: Vec<f64> = _z_bounds.iter().map(|f| f.as_f64().unwrap()).collect();

        let energy_mesh = deterministicinput["energy_bounds"].as_array().unwrap();
        let energetic_mesh: Vec<f64> = energy_mesh.iter().map(|f| f.as_f64().unwrap()).collect();

        let mesh = common::cartesian_mesh::CartesianMesh {
            x_bounds: x_bounds,
            y_bounds: y_bounds,
            z_bounds: z_bounds,
            energy_bounds: energetic_mesh,
            tallies: Array4::<Tally>::zeros((1, 1, 1, 1)),
        };
        if _solver_type.eq("finite_differences") {
            deterministic_solver = Some(FiniteDifferencesSolver::new(
                    calculation_mode,
                    ct,
                    mesh));
        } else {
            panic!(
                "solver_type for deterministic calculation not recognized: {}",
                _solver_type
            );
        }
    }

    if deterministic_solver.is_some() {
        deterministic_solver.unwrap().solve(
            &mut output_file,
            &geometry,
            &cross_section_library,
            &sources,
            &mut detectors,
            &mut meshes,
        )
    }
    Ok(())
}
