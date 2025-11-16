use crate::chess::{Board, Move, MoveList, PieceKind, piece_value};
use crate::nnue_runtime::NnueRuntime;
use crate::time_manager::TimeBudget;
use crate::tt::{Bound, TranspositionTable};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

const MATE_VALUE: i32 = 30_000;
const MAX_PLY: usize = 64;

#[derive(Debug, Default, Clone, Copy)]
pub struct SearchStatistics {
    pub nodes: u64,
}

#[derive(Clone)]
struct SearchState {
    killer_moves: [[Option<Move>; 2]; MAX_PLY],
    history: [[u32; 64]; 64],
    optimal_deadline: Option<Instant>,
    hard_deadline: Option<Instant>,
    abort_flag: Option<Arc<AtomicBool>>,
    stopped: bool,
}

fn move_indices(mv: Move) -> (usize, usize) {
    let from = mv.from.rank() as usize * 8 + mv.from.file() as usize;
    let to = mv.to.rank() as usize * 8 + mv.to.file() as usize;
    (from, to)
}

impl SearchState {
    fn new(
        deadlines: Option<(Instant, Instant)>,
        abort_flag: Option<Arc<AtomicBool>>,
    ) -> Self {
        Self {
            killer_moves: [[None; 2]; MAX_PLY],
            history: [[0; 64]; 64],
            optimal_deadline: deadlines.map(|d| d.0),
            hard_deadline: deadlines.map(|d| d.1),
            abort_flag,
            stopped: false,
        }
    }

    fn killer_moves(&self, ply: usize) -> [Option<Move>; 2] {
        self.killer_moves[ply.min(MAX_PLY - 1)]
    }

    fn record_killer(&mut self, ply: usize, mv: Move) {
        let idx = ply.min(MAX_PLY - 1);
        if self.killer_moves[idx].contains(&Some(mv)) {
            return;
        }
        self.killer_moves[idx][1] = self.killer_moves[idx][0];
        self.killer_moves[idx][0] = Some(mv);
    }

    fn record_history(&mut self, mv: Move, depth: u8) {
        let (from, to) = move_indices(mv);
        const HISTORY_LIMIT: u32 = 1_000_000;
        let bonus = (depth as u32) * (depth as u32);
        let entry = &mut self.history[from][to];
        *entry = entry.saturating_add(bonus).min(HISTORY_LIMIT);
    }

    fn history_score(&self, mv: Move) -> i32 {
        let (from, to) = move_indices(mv);
        self.history[from][to] as i32
    }

    fn should_stop(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        if let Some(flag) = &self.abort_flag {
            if flag.load(Ordering::Relaxed) {
                self.stopped = true;
                return true;
            }
        }
        let now = Instant::now();
        if let Some(opt_deadline) = self.optimal_deadline {
            if now >= opt_deadline {
                self.stopped = true;
                return true;
            }
        }
        if let Some(hard_deadline) = self.hard_deadline {
            if now >= hard_deadline {
                self.stopped = true;
                return true;
            }
        }
        false
    }
}

#[derive(Clone, Copy)]
struct SearchWindow {
    alpha: i32,
    beta: i32,
}

impl SearchWindow {
    fn new(alpha: i32, beta: i32) -> Self {
        Self { alpha, beta }
    }

    fn flipped(self) -> Self {
        SearchWindow::new(-self.beta, -self.alpha)
    }
}

struct SearchContext<'a> {
    board: &'a mut Board,
    tt: &'a mut TranspositionTable,
    stats: &'a mut SearchStatistics,
    state: &'a mut SearchState,
    nnue_runner: &'a NnueRuntime,
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
    board: &mut Board,
    tt: &mut TranspositionTable,
    max_depth: u8,
    time_budget: Option<TimeBudget>,
    nnue_runner: &NnueRuntime,
    abort_flag: Option<Arc<AtomicBool>>,
) -> SearchReport {
    let mut stats = SearchStatistics::default();
    let mut root_moves = MoveList::new();
    board.legal_moves_into(&mut root_moves);
    if root_moves.is_empty() {
        return SearchReport {
            best_move: None,
            depth: 0,
            stats: SearchStatistics::default(),
            elapsed: Duration::from_millis(0),
        };
    }

    let start = Instant::now();
    let deadlines = time_budget.map(|b| (start + b.optimal, start + b.maximum));
    let mut state = SearchState::new(deadlines, abort_flag.clone());

    let mut best_move = None;
    let mut completed_depth = 0;
    let mut last_score = 0;
    const ASP_WINDOW: i32 = 50;

    {
        let mut ctx = SearchContext {
            board,
            tt,
            stats: &mut stats,
            state: &mut state,
            nnue_runner,
        };

        'search: for current_depth in 1..=max_depth {
            if ctx.state.should_stop() {
                break;
            }
            let mut window = if current_depth > 1 {
                SearchWindow::new(
                    (last_score - ASP_WINDOW).max(-MATE_VALUE),
                    (last_score + ASP_WINDOW).min(MATE_VALUE),
                )
            } else {
                SearchWindow::new(-MATE_VALUE, MATE_VALUE)
            };

            loop {
                let (candidate_move, score) = search_single_depth(
                    &mut ctx,
                    current_depth,
                    &mut root_moves,
                    best_move,
                    window,
                );
                if ctx.state.should_stop() {
                    break 'search;
                }

                if score <= window.alpha && window.alpha > -MATE_VALUE {
                    window.alpha = window.alpha.saturating_sub(ASP_WINDOW * 2);
                    continue;
                } else if score >= window.beta && window.beta < MATE_VALUE {
                    window.beta = window.beta.saturating_add(ASP_WINDOW * 2);
                    continue;
                } else {
                    best_move = candidate_move;
                    last_score = score;
                    completed_depth = current_depth;
                    break;
                }
            }
        }
    }

    SearchReport {
        best_move,
        depth: completed_depth,
        stats,
        elapsed: start.elapsed(),
    }
}

fn search_single_depth(
    ctx: &mut SearchContext,
    depth: u8,
    root_moves: &mut MoveList,
    preferred: Option<Move>,
    window: SearchWindow,
) -> (Option<Move>, i32) {
    let tt_move = ctx.tt.probe(ctx.board.hash()).and_then(|entry| entry.best_move);
    let killers = ctx.state.killer_moves(0);
    order_moves(ctx.board, root_moves, preferred, tt_move, killers, ctx.state);

    let mut best_move = None;
    let mut best_score = -MATE_VALUE;
    let mut bounds = window;

    for &mv in root_moves.iter() {
        let undo = ctx.board.make_move(mv);
        let score = -negamax(
            ctx,
            depth.saturating_sub(1),
            SearchWindow::new(-bounds.beta, -bounds.alpha),
            1,
        );
        ctx.board.unmake_move(undo);
        if score > best_score {
            best_score = score;
            best_move = Some(mv);
        }
        bounds.alpha = bounds.alpha.max(best_score);
        if bounds.alpha >= bounds.beta {
            break;
        }
    }

    (best_move, best_score)
}

fn negamax(
    ctx: &mut SearchContext,
    depth: u8,
    mut window: SearchWindow,
    ply: usize,
) -> i32 {
    if ctx.state.should_stop() {
        return 0;
    }
    ctx.stats.record_node();

    let alpha_orig = window.alpha;
    let beta_orig = window.beta;

    let key = ctx.board.hash();
    let mut tt_move = None;
    if let Some(entry) = ctx.tt.probe(key) {
        tt_move = entry.best_move;
        if entry.depth >= depth {
            match entry.bound {
                Bound::Exact => return entry.value,
                Bound::Lower => window.alpha = window.alpha.max(entry.value),
                Bound::Upper => window.beta = window.beta.min(entry.value),
            }
            if window.alpha >= window.beta {
                return entry.value;
            }
        }
    }

    if depth == 0 {
        return quiescence(ctx, window, ply);
    }

    const NULL_MOVE_REDUCTION: u8 = 2;
    if depth > NULL_MOVE_REDUCTION + 1 && !ctx.board.is_in_check(ctx.board.active_color)
    {
        let undo = ctx.board.make_null_move();
        let score = -negamax(
            ctx,
            depth - 1 - NULL_MOVE_REDUCTION,
            SearchWindow::new(-window.beta, -window.beta + 1),
            ply + 1,
        );
        ctx.board.unmake_null_move(undo);
        if score >= window.beta {
            return score;
        }
    }

    let mut moves = MoveList::new();
    ctx.board.legal_moves_into(&mut moves);
    let killers = ctx.state.killer_moves(ply);
    order_moves(ctx.board, &mut moves, None, tt_move, killers, ctx.state);
    if moves.is_empty() {
        if ctx.board.is_in_check(ctx.board.active_color) {
            return -MATE_VALUE + depth as i32;
        } else {
            return 0;
        }
    }

    let mut best_value = -MATE_VALUE;
    let mut best_move = None;
    for &mv in moves.as_slice() {
        let undo = ctx.board.make_move(mv);
        let is_capture = undo.captured_piece.is_some();
        if is_capture && depth > 0 && ctx.board.static_exchange_eval(mv) < 0 {
            ctx.board.unmake_move(undo);
            continue;
        }
        let score = -negamax(
            ctx,
            depth.saturating_sub(1),
            window.flipped(),
            ply + 1,
        );
        ctx.board.unmake_move(undo);
        if score > best_value {
            best_value = score;
            best_move = Some(mv);
        }
        window.alpha = window.alpha.max(score);
        if window.alpha >= window.beta {
            if !is_capture {
                ctx.state.record_killer(ply, mv);
                ctx.state.record_history(mv, depth);
            }
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
    ctx.tt.store(key, depth, best_value, bound, best_move);
    best_value
}

fn quiescence(ctx: &mut SearchContext, mut window: SearchWindow, ply: usize) -> i32 {
    if ctx.state.should_stop() {
        return window.alpha;
    }
    ctx.stats.record_node();

    let stand_pat = ctx.nnue_runner.eval(ctx.board).unwrap_or(-MATE_VALUE);
    if stand_pat >= window.beta {
        return stand_pat;
    }
    if window.alpha < stand_pat {
        window.alpha = stand_pat;
    }

    let mut moves = MoveList::new();
    ctx.board.legal_moves_into(&mut moves);
    let tt_move = ctx.tt.probe(ctx.board.hash()).and_then(|entry| entry.best_move);
    let killers = ctx.state.killer_moves(ply);
    order_moves(ctx.board, &mut moves, None, tt_move, killers, ctx.state);

    for &mv in moves.as_slice() {
        if ctx.board.piece_at(mv.to).is_none() {
            continue;
        }
        if ctx.board.static_exchange_eval(mv) < 0 {
            continue;
        }
        let undo = ctx.board.make_move(mv);
        let score = -quiescence(
            ctx,
            SearchWindow::new(-window.beta, -window.alpha),
            ply + 1,
        );
        ctx.board.unmake_move(undo);

        if score >= window.beta {
            return window.beta;
        }
        window.alpha = window.alpha.max(score);
    }

    window.alpha
}

fn order_moves(
    board: &Board,
    moves: &mut MoveList,
    preferred: Option<Move>,
    tt_move: Option<Move>,
    killer_moves: [Option<Move>; 2],
    state: &SearchState,
) {
    let slice = moves.as_mut_slice();
    slice.sort_by_key(|mv| {
        move_score(board, *mv, preferred, tt_move, killer_moves, state)
    });
    slice.reverse();
}

fn move_score(
    board: &Board,
    mv: Move,
    preferred: Option<Move>,
    tt_move: Option<Move>,
    killer_moves: [Option<Move>; 2],
    state: &SearchState,
) -> i32 {
    const PREFERRED_BONUS: i32 = 1_000_000;
    const PROMOTION_BONUS: i32 = 50_000;
    const CAPTURE_BONUS: i32 = 10_000;
    const TT_BONUS: i32 = 2_000_000;
    const KILLER_BONUS: i32 = 30_000;

    let mut score = 0;
    if tt_move == Some(mv) {
        score += TT_BONUS;
    }

    if preferred == Some(mv) {
        score += PREFERRED_BONUS;
    }

    if let Some(killer) = killer_moves[0] {
        if killer == mv {
            score += KILLER_BONUS;
        }
    }
    if let Some(killer) = killer_moves[1] {
        if killer == mv {
            score += KILLER_BONUS / 2;
        }
    }

    if let Some(moving) = board.piece_at(mv.from) {
        if moving.kind == PieceKind::Pawn && (mv.to.rank() == 0 || mv.to.rank() == 7)
        {
            score += PROMOTION_BONUS;
        }
        if let Some(captured) = board.piece_at(mv.to) {
            score +=
                CAPTURE_BONUS + piece_value(captured.kind) - piece_value(moving.kind);
        }
    }

    score += state.history_score(mv);

    score
}
