use crate::chess::{Board, Move};
use crate::nnue::NnueRunner;
use crate::tt::{Bound, TranspositionTable};
use std::time::{Duration, Instant};

const MATE_VALUE: i32 = 30_000;

#[derive(Debug, Default, Clone, Copy)]
pub struct SearchStatistics {
    pub nodes: u64,
}

impl SearchStatistics {
    fn record_node(&mut self) {
        self.nodes += 1;
    }
}

#[derive(Debug, Clone)]
pub struct SearchReport {
    pub best_move: Option<Move>,
    pub depth: u8,
    pub stats: SearchStatistics,
    pub elapsed: Duration,
}

impl SearchReport {
    pub fn nps(&self) -> u64 {
        let elapsed_ms = self.elapsed.as_millis() as u64;
        if elapsed_ms == 0 {
            self.stats.nodes
        } else {
            (self.stats.nodes * 1000) / elapsed_ms.max(1)
        }
    }
}

pub fn search_best_move(
    board: &Board,
    tt: &mut TranspositionTable,
    depth: u8,
    nnue_runner: &NnueRunner,
) -> SearchReport {
    let moves = board.legal_moves();
    if moves.is_empty() {
        return SearchReport {
            best_move: None,
            depth,
            stats: SearchStatistics::default(),
            elapsed: Duration::from_millis(0),
        };
    }

    let start = Instant::now();
    let mut best_move = None;
    let mut best_score = -MATE_VALUE;
    let mut alpha = -MATE_VALUE;
    let beta = MATE_VALUE;
    let mut stats = SearchStatistics::default();

    for mv in moves {
        let mut next = board.clone();
        next.apply_move_unchecked(mv);
        let score = -negamax(
            &next,
            tt,
            depth.saturating_sub(1),
            -beta,
            -alpha,
            &mut stats,
            nnue_runner,
        );
        if score > best_score {
            best_score = score;
            best_move = Some(mv);
        }
        alpha = alpha.max(best_score);
    }

    SearchReport {
        best_move,
        depth,
        stats,
        elapsed: start.elapsed(),
    }
}

fn negamax(
    board: &Board,
    tt: &mut TranspositionTable,
    depth: u8,
    mut alpha: i32,
    mut beta: i32,
    stats: &mut SearchStatistics,
    nnue_runner: &NnueRunner,
) -> i32 {
    stats.record_node();

    let alpha_orig = alpha;
    let beta_orig = beta;

    let key = board.hash();
    if let Some(entry) = tt.probe(key)
        && entry.depth >= depth
    {
        match entry.bound {
            Bound::Exact => return entry.value,
            Bound::Lower => alpha = alpha.max(entry.value),
            Bound::Upper => beta = beta.min(entry.value),
        }
        if alpha >= beta {
            return entry.value;
        }
    }

    if depth == 0 {
        let eval = nnue_runner.eval(board).unwrap_or(-MATE_VALUE);
        tt.store(key, depth, eval, Bound::Exact);
        return eval;
    }

    let moves = board.legal_moves();
    if moves.is_empty() {
        if board.is_in_check(board.active_color) {
            return -MATE_VALUE + depth as i32;
        } else {
            return 0;
        }
    }

    let mut best_value = -MATE_VALUE;
    for mv in moves {
        let mut child = board.clone();
        child.apply_move_unchecked(mv);
        let score = -negamax(
            &child,
            tt,
            depth.saturating_sub(1),
            -beta,
            -alpha,
            stats,
            nnue_runner,
        );
        best_value = best_value.max(score);
        alpha = alpha.max(score);
        if alpha >= beta {
            break;
        }
    }

    let bound = if best_value <= alpha_orig {
        Bound::Upper
    } else if best_value >= beta_orig {
        Bound::Lower
    } else {
        Bound::Exact
    };
    tt.store(key, depth, best_value, bound);
    best_value
}
