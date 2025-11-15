use suckfish::chess::Board;
use suckfish::nnue_runtime::NnueRuntime;
use suckfish::search::search_best_move;
use suckfish::tt::TranspositionTable;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
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
    let nnue_runner = NnueRuntime::new(cmd_args.nnue_path)?;
    let mut tt = TranspositionTable::new(16);
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
                let time_ms = time_left.trim().parse::<u64>().unwrap_or(0);
                let think_time = if time_ms > 0 {
                    // Spend a small fraction of the remaining time.
                    let slice = (time_ms / 30).max(10);
                    Some(Duration::from_millis(slice))
                } else {
                    None
                };
                let sr = search_best_move(
                    &mut board,
                    &mut tt,
                    8,
                    think_time,
                    &nnue_runner,
                );
                eprintln!("{:?}, nps: {}", sr, sr.nps());
                if let Some(bm) = sr.best_move {
                    println!("{}", bm.to_uci());
                }
            }
            Some("newgame") => {
                tt.clear();
                println!("newgame ready");
            }
            _ => break,
        }
    }
    Ok(())
}
