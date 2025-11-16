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
    mpsc::{self, Receiver, TryRecvError},
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
    let time_manager = TimeManager::default();
    loop {
        let mut cmdline = String::new();
        std::io::stdin().read_line(&mut cmdline)?;
        let mut cmd_parsed = cmdline.split(" ");
        let cmd = cmd_parsed.next();
        match cmd {
            Some("go") => {
                let time_left = if let Some(tl) = cmd_parsed.next() {
                    tl.to_string()
                } else {
                    continue;
                };
                let fen = &cmdline[(time_left.len() + 4)..];
                let mut board = Board::from_fen(fen)?;
                let position_hash = board.hash();

                if let Some(mut handle) = ponder.take() {
                    if handle.target_hash == position_hash {
                        if let Some(sr) = handle.try_take_result() {
                            eprintln!("{:?}, nps: {}", sr, sr.nps());
                            if let Some(bm) = sr.best_move {
                                println!("{}", bm.to_uci());
                                ponder =
                                    start_ponder(&board, bm, nnue_runner.clone());
                            }
                            continue;
                        }
                    }
                    handle.stop();
                }

                let time_ms = time_left.trim().parse::<u64>().unwrap_or(0);
                let think_time = time_manager.compute_budget(time_ms);
                let sr = search_best_move(
                    &mut board,
                    &mut tt,
                    32,
                    think_time,
                    nnue_runner.as_ref(),
                    None,
                );
                eprintln!("{:?}, nps: {}", sr, sr.nps());
                if let Some(bm) = sr.best_move {
                    println!("{}", bm.to_uci());
                    ponder = start_ponder(&board, bm, nnue_runner.clone());
                }
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
    fn stop(mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    fn try_take_result(&mut self) -> Option<SearchReport> {
        match self.result_rx.try_recv() {
            Ok(sr) => {
                if let Some(handle) = self.handle.take() {
                    let _ = handle.join();
                }
                Some(sr)
            }
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => {
                if let Some(handle) = self.handle.take() {
                    let _ = handle.join();
                }
                None
            }
        }
    }
}

fn start_ponder(
    current_board: &Board,
    our_move: Move,
    nnue: Arc<NnueRuntime>,
) -> Option<PonderHandle> {
    let mut board_after = current_board.clone();
    if !board_after.play_move(our_move) {
        return None;
    }

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
    );
    let reply = reply_report.best_move?;
    if !board_after.play_move(reply) {
        return None;
    }

    let target_hash = board_after.hash();
    let stop_flag = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let mut ponder_board = board_after;
    let mut ponder_tt = TranspositionTable::new(16);
    let nnue_clone = nnue.clone();
    let stop_clone = stop_flag.clone();

    let ponder_budget = TimeBudget {
        optimal: Duration::from_millis(1000),
        maximum: Duration::from_millis(3000),
    };
    let handle = thread::spawn(move || {
        let report = search_best_move(
            &mut ponder_board,
            &mut ponder_tt,
            8,
            Some(ponder_budget),
            nnue_clone.as_ref(),
            Some(stop_clone),
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
