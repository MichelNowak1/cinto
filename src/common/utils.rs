use crate::common::cross_section_library::EvaluationType;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::Rng;

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub enum Monotony {
    Increasing,
    Decreasing,
}

pub enum DistributionType {
    Discrete,
    Continuous,
}

/// Monotony of vector
pub fn get_monotony(values: &Vec<f64>) -> Monotony {
    let len = values.len();

    if values[0] > values[len - 1] {
        return Monotony::Decreasing;
    } else if values[0] < values[len - 1] {
        return Monotony::Increasing;
    }
    panic!("Cannot get monotony of vector.");
}

/// Search function
/// values must be ordered from min to max, or from max to min
///
/// # Arguments
/// * `values` vector containing the values in which we wish to find the index
pub fn get_index(value: f64, values: &Vec<f64>) -> Option<usize> {
    let len = values.len();
    let monotony = get_monotony(values);

    // first, check if value is in reachable range
    match monotony {
        Monotony::Increasing => {
            if value < values[0] ||
               value > values[len - 1] {
                return None;
            }
        }
        Monotony::Decreasing => {
            if value < values[len - 1] ||
               value > values[0] {
                return None;
            }
        }
    }
    if (value - values[0]).abs() < f64::EPSILON {
        return Some(0)
    }
    if (value - values[len-1]).abs() < f64::EPSILON  {
        return Some(len-2)
    }
    let index = match get_monotony(values) {
        Monotony::Increasing => {
            values.iter().position(|v| value < *v)
        },
        Monotony::Decreasing=> {
            values.iter().position(|v| value > *v)
        }
    };
    if index.is_some() {
        Some(index.unwrap()-1)
    } else {
        None
    }
}

pub fn weighted_sample(values: &Vec<f64>) -> usize {
    let mut rng = rand::thread_rng();
    let dist = WeightedIndex::new(values).unwrap();
    let choosen_index: usize = dist.sample(&mut rng);
    choosen_index
}

pub fn get_interpolated_value(
    value: f64,
    values: &Vec<f64>,
    y: &Vec<f64>
    ) -> f64 {

    let index_ = get_index(value, &values);
    if !index_.is_some() {
        panic!("could not interpolate value {} in {:?}", value, values);
    }
    let index = index_.unwrap();

    let mut library_type = EvaluationType::Punctual;
    if values.len() - 1 == y.len() {
        library_type = EvaluationType::Multigroup;
    }

    match library_type {
        EvaluationType::Punctual => {
            (y[index + 1] - y[index]) /
                (values[index + 1] - values[index])
                * (value - values[index])
                + y[index]
        }
        EvaluationType::Multigroup => y[index],
    }
}

/// Sample from discrete distribution
///
/// # Arguments
/// * `elements` discrete set of elements in which one needs to be sampled
/// * `probability_distribution` discrete probability distribution associated to elements
///
/// # Returns
/// * `element` returns a reference to the sampled element
///
/// # Example
/// ```
/// let interactions = [
///     common::cross_section::Interaction::Fission,
///     common::cross_section::Interaction::ElasticScattering,
///     common::cross_section::Interaction::InelasticScattering,
/// ];
///
/// let probability_distribution = [0.1, 0.6, 0.3];
/// let sampled_interaction = sample_from_discrete_distribution(
///     &interactions,
///     probability_distribution);
/// ```
pub fn sample_from_discrete_distribution<T: Clone>(
    elements: &Vec<T>,
    probability_distribution: Vec<f64>,
) -> T {
    // fill the probability table
    let mut cumulative_distribution = vec![0.];
    let mut cumulative_distribution_norm = 0.;

    for index in 0..probability_distribution.len() {
        cumulative_distribution_norm += probability_distribution[index];
        cumulative_distribution.push(cumulative_distribution_norm);
    }

    let mut rng = rand::thread_rng();
    let mut xsi: f64 = rng.r#gen();
    xsi = cumulative_distribution_norm * xsi;

    let index: Option<usize> =
        if cumulative_distribution_norm < 1E-10 {
        Some((xsi * (elements.len() as f64)) as usize)
    } else {
        get_index(xsi, &cumulative_distribution)
    };

    elements[index.unwrap()].clone()
}

pub fn sample_from_distribution(
    x: &Vec<f64>,
    probability_distribution: &Vec<f64>
    ) -> f64 {

    let dim = probability_distribution.len();
    let mut distribution_type = DistributionType::Discrete;
    if x.len() == dim {
        distribution_type = DistributionType::Continuous;
    }

    match distribution_type {
        DistributionType::Discrete => {
            let mut cumulated_distribution = vec![0.; dim + 1];
            let mut norm = 0.;

            for index in 1..dim + 1 {
                norm += probability_distribution[index - 1] *
                    (x[index] - x[index - 1]).abs();
                cumulated_distribution[index] = norm;
            }

            let mut rng = rand::thread_rng();
            let mut xsi: f64 = rng.r#gen();
            xsi = norm * xsi;

            get_interpolated_value(
                xsi,
                &cumulated_distribution,
                &x[..dim].to_vec())
        }

        DistributionType::Continuous => {
            let mut cumulated_distribution = vec![0.; dim];
            let mut norm = 0.;

            for index in 1..dim {
                norm += (probability_distribution[index] +
                         probability_distribution[index - 1])
                    * (x[index] - x[index - 1])
                    / 2.;
                cumulated_distribution[index] = norm;
            }

            let mut rng = rand::thread_rng();
            let mut xsi: f64 = rng.r#gen();
            xsi = norm * xsi;

            get_interpolated_value(
                xsi,
                &cumulated_distribution,
                &x)
        }
    }
}
