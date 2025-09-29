use indicatif::{ProgressBar, ProgressStyle};
use std::time::Instant;

pub struct Profiler {
    pub start: Instant,
}

impl Profiler {
    pub fn new() -> Self {
        Profiler {
            start: Instant::now(),
        }
    }
    pub fn reset(&mut self) {
        self.start = Instant::now();
    }
}

pub struct CintoProgressBar {
    pub progress_bar: ProgressBar,
}

impl CintoProgressBar {
    pub fn new(size: usize) -> Self {
        let sty = ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}");

        let progress_bar = ProgressBar::new(size as u64);
        progress_bar.set_style(sty.unwrap());

        CintoProgressBar {
            progress_bar: progress_bar,
        }
    }
    pub fn update(&mut self, index: u64) {
        self.progress_bar.set_message(format!("batch #{}", index));
        self.progress_bar.inc(1);
    }
    pub fn finish(&mut self) {
        self.progress_bar.finish_with_message("done");
    }
}
