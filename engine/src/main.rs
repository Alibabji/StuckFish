use suckfish::chess::{Board, Move};
use suckfish::nnue_runtime::NnueRuntime;
use suckfish::search::{search_best_move, SearchReport};
use suckfish::time_manager::{TimeBudget, TimeManager};
use suckfish::tt::TranspositionTable;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{self, Receiver},
    Arc,
};
use std::thread;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(about = "Suckfish: a homemade chess engine", author, version)]
struct CmdArgs {
    /// NNUE weight file
    #[arg(long, value_name = "NNUE_PATH", default_value = "nnue.ot")]
    nnue_path: PathBuf,
}

fn main() -> Result<()> {
    let cmd_args = CmdArgs::parse();
    let nnue_runner = Arc::new(NnueRuntime::new(cmd_args.nnue_path)?);
    let mut tt = TranspositionTable::new(16);
    let mut ponder: Option<PonderHandle> = None;
    let mut history = GameHistory::new();
    let time_manager = TimeManager::default();
    loop {
        let mut cmdline = String::new();
        if std::io::stdin().read_line(&mut cmdline)? == 0 {
            break;
        }
        let trimmed = cmdline.trim_end();
        if trimmed.is_empty() {
            continue;
        }

        let mut idx = 0;
        let bytes = trimmed.as_bytes();
        while idx < bytes.len() && !bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        let cmd = &trimmed[..idx];
        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        let time_start = idx;
        while idx < bytes.len() && !bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        let time_slice = &trimmed[time_start..idx];
        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        let fen_slice = if idx < bytes.len() {
            &trimmed[idx..]
        } else {
            ""
        };

        match cmd {
            "go" => {
                if fen_slice.is_empty() {
                    continue;
                }
                let time_left = if time_slice.is_empty() {
                    "0"
                } else {
                    time_slice
                };
                let fen = fen_slice;
                let mut board = Board::from_fen(fen)?;
                history.reset(&board);
                let position_hash = board.hash();

                let time_ms = time_left.trim().parse::<u64>().unwrap_or(0);
                let default_budget = TimeBudget {
                    optimal: Duration::from_millis(100),
                    maximum: Duration::from_millis(300),
                };
                let think_time = time_manager
                    .compute_budget(time_ms)
                    .unwrap_or(default_budget);
                if let Some(handle) = ponder.take() {
                    if handle.target_hash == position_hash {
                        if let Some(ponder_sr) = handle.take_result() {
                            let remaining_opt = think_time
                                .optimal
                                .checked_sub(ponder_sr.elapsed)
                                .unwrap_or_else(Duration::default);
                            let remaining_max = think_time
                                .maximum
                                .checked_sub(ponder_sr.elapsed)
                                .unwrap_or_else(Duration::default);
                            let needs_search =
                                remaining_max > Duration::from_millis(10);
                            let final_report = if needs_search {
                                let refined_budget = TimeBudget {
                                    optimal: remaining_opt
                                        .max(Duration::from_millis(1)),
                                    maximum: remaining_max
                                        .max(Duration::from_millis(1)),
                                };
                                let refined = search_best_move(
                                    &mut board,
                                    &mut tt,
                                    32,
                                    Some(refined_budget),
                                    nnue_runner.as_ref(),
                                    None,
                                    history.as_slice(),
                                );
                                if refined.depth >= ponder_sr.depth {
                                    refined
                                } else {
                                    ponder_sr
                                }
                            } else {
                                ponder_sr
                            };
                            eprintln!(
                                "{:?}, nps: {}",
                                final_report,
                                final_report.nps()
                            );
                            if let Some(bm) = final_report.best_move {
                                println!("{}", bm.to_uci());
                                ponder = start_ponder(
                                    &board,
                                    bm,
                                    nnue_runner.clone(),
                                    history.as_slice(),
                                );
                            }
                            continue;
                        }
                    } else {
                        handle.abort();
                    }
                }

                let sr = search_best_move(
                    &mut board,
                    &mut tt,
                    32,
                    Some(think_time),
                    nnue_runner.as_ref(),
                    None,
                    history.as_slice(),
                );
                eprintln!("{:?}, nps: {}", sr, sr.nps());
                if let Some(bm) = sr.best_move {
                    println!("{}", bm.to_uci());
                    ponder = start_ponder(
                        &board,
                        bm,
                        nnue_runner.clone(),
                        history.as_slice(),
                    );
                }
            }
            "newgame" => {
                if let Some(handle) = ponder.take() {
                    handle.abort();
                }
                tt.clear();
                history.clear();
                println!("newgame ready");
            }
            _ => break,
        }
    }
    Ok(())
}

struct PonderHandle {
    target_hash: u64,
    stop_flag: Arc<AtomicBool>,
    result_rx: Receiver<SearchReport>,
    handle: Option<thread::JoinHandle<()>>,
}

impl PonderHandle {
    fn take_result(mut self) -> Option<SearchReport> {
        if let Ok(sr) = self.result_rx.try_recv() {
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
            return Some(sr);
        }
        self.stop_flag.store(true, Ordering::Relaxed);
        let result = self.result_rx.recv().ok();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        result
    }

    fn abort(mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn start_ponder(
    current_board: &Board,
    our_move: Move,
    nnue: Arc<NnueRuntime>,
    history: &[u64],
) -> Option<PonderHandle> {
    let mut board_after = current_board.clone();
    if !board_after.play_move(our_move) {
        return None;
    }

    let mut ponder_history = history.to_vec();
    append_history(&mut ponder_history, &board_after);

    let mut reply_board = board_after.clone();
    let mut tmp_tt = TranspositionTable::new(16);
    let quick_budget = TimeBudget {
        optimal: Duration::from_millis(50),
        maximum: Duration::from_millis(100),
    };
    let reply_report = search_best_move(
        &mut reply_board,
        &mut tmp_tt,
        1,
        Some(quick_budget),
        nnue.as_ref(),
        None,
        &ponder_history,
    );
    let reply = reply_report.best_move?;
    if !board_after.play_move(reply) {
        return None;
    }
    append_history(&mut ponder_history, &board_after);

    let target_hash = board_after.hash();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let mut ponder_board = board_after;
    let mut ponder_tt = TranspositionTable::new(16);
    let nnue_clone = nnue.clone();
    let stop_clone = stop_flag.clone();

    let ponder_budget = TimeBudget {
        optimal: Duration::from_secs(20),
        maximum: Duration::from_secs(20),
    };
    let handle = thread::spawn(move || {
        let report = search_best_move(
            &mut ponder_board,
            &mut ponder_tt,
            8,
            Some(ponder_budget),
            nnue_clone.as_ref(),
            Some(stop_clone),
            &ponder_history,
        );
        let _ = tx.send(report);
    });

    Some(PonderHandle {
        target_hash,
        stop_flag,
        result_rx: rx,
        handle: Some(handle),
    })
}

fn append_history(history: &mut Vec<u64>, board: &Board) {
    if board.halfmove_clock == 0 {
        history.clear();
    }
    history.push(board.hash());
    let max_len = board.halfmove_clock as usize + 1;
    if history.len() > max_len {
        let remove = history.len() - max_len;
        history.drain(0..remove);
    }
}

struct GameHistory {
    positions: Vec<u64>,
}

impl GameHistory {
    fn new() -> Self {
        Self {
            positions: Vec::new(),
        }
    }

    fn reset(&mut self, board: &Board) {
        self.positions.clear();
        append_history(&mut self.positions, board);
    }

    fn clear(&mut self) {
        self.positions.clear();
    }

    fn as_slice(&self) -> &[u64] {
        &self.positions
    }
}
