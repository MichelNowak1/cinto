use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

use crate::common::importance_map::importance::Importance;

#[derive(PartialEq, Eq, Hash)]
pub enum Phase {
    Explore,
    Exploit,
    ExploreExploit,
}

#[derive(PartialEq, Eq, Hash)]
pub enum Scheme {
    Alternate,
    Bootstrap,
}

/// Strategy to adopt to iterate on variance reduction techniques
pub struct Strategy {
    pub scheme: Scheme,
    pub current_phase: Phase,
    pub phases: Vec<Phase>,
    pub phase_length: HashMap<Phase, u32>,
    pub batch_index_at_last_update: u32,
}

impl Strategy {
    /// Ends batch by managing importances according to strategy
    ///
    /// # Arguments
    /// * `importance` current importance map used for transport
    /// * `scored_importance` current importance map scored during transport
    /// * `scoring_importance` boolean that tells if next batch should be scoring importance
    /// * `batch_index` current batch index
    pub fn end_batch(
        &mut self,
        importance: &mut Rc<RefCell<dyn Importance>>,
        scored_importance: &mut Rc<RefCell<dyn Importance>>,
        scoring_importance: &mut bool,
        batch_index_: u32,
    ) {
        // Leave loop if phase if ExploreExploit, because there is nothing to do anymore
        if self.current_phase == Phase::ExploreExploit {
            return;
        }

        // Compute number of batches since last update
        let cumulated_batches_in_current_phase_ =
            batch_index_ - self.batch_index_at_last_update;

        // Do not update anything if number of batch since last update is not equal to
        // the length of the current phase
        if cumulated_batches_in_current_phase_ !=
            self.phase_length[&self.current_phase] {
            return;
        }

        // update batch at which the last update has been made
        self.batch_index_at_last_update = batch_index_;

        // Alternate scheme
        if self.scheme == Scheme::Alternate {
            // Alternate exploration and exploitation phases
            if self.current_phase == Phase::Explore {
                self.current_phase = Phase::Exploit;
                *scoring_importance = false;
            } else if self.current_phase == Phase::Exploit {
                self.current_phase = Phase::Explore;
                *scoring_importance = true;
            }

            // Swap importances
            std::mem::swap(importance, scored_importance);

        // Bootstrap scheme
        } else if self.scheme == Scheme::Bootstrap {
            // Swap importances at the end of the first exploration phase
            if batch_index_ == self.phase_length[&Phase::Explore] {
                self.current_phase = Phase::ExploreExploit;

                // Now, scored importance is also the importance
                *importance = scored_importance.clone();
            }

            // Continue scoring importance all along the simulation
            *scoring_importance = true;
        }
    }

    /// Returns current phase as string
    ///
    /// # `current_phase_string` string to be printed while managing results
    pub fn get_current_phase_as_string(&self) -> String {
        if self.current_phase == Phase::Explore {
            return "exploration".to_string();
        } else if self.current_phase == Phase::Exploit {
            return "exploitation".to_string();
        } else if self.current_phase == Phase::ExploreExploit {
            return "exploration_exploitation".to_string();
        }
        panic!("unknown phase");
    }
}
