use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct TimeBudget {
    pub optimal: Duration,
    pub maximum: Duration,
}

pub struct TimeManager {
    safety_ratio: f64,
    max_ratio: f64,
    avg_moves: u32,
    min_time: Duration,
}

impl Default for TimeManager {
    fn default() -> Self {
        Self {
            safety_ratio: 0.02,
            max_ratio: 0.5,
            avg_moves: 40,
            min_time: Duration::from_millis(10),
        }
    }
}

impl TimeManager {
    pub fn compute_budget(&self, millis_left: u64) -> Option<TimeBudget> {
        if millis_left == 0 {
            return None;
        }
        let time_left = millis_left as f64;
        let reserve = (time_left * self.safety_ratio).min(time_left / 4.0);
        let usable = (time_left - reserve).max(0.0);
        let base = usable / self.avg_moves as f64;
        let optimal_ms = base.max(self.min_time.as_millis() as f64);
        let maximum_ms = (optimal_ms * 3.0).min(time_left * self.max_ratio);
        Some(TimeBudget {
            optimal: Duration::from_millis(optimal_ms as u64),
            maximum: Duration::from_millis(maximum_ms as u64),
        })
    }

    pub fn deadlines(
        &self,
        start: Instant,
        budget: &TimeBudget,
    ) -> (Instant, Instant) {
        (start + budget.optimal, start + budget.maximum)
    }
}
