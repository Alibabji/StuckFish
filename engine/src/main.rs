use suckfish::chess::Board;
use suckfish::nnue::NnueRunner;
use suckfish::search::search_best_move;
use suckfish::tt::TranspositionTable;

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tch::Device;

#[derive(Parser, Debug)]
#[command(about = "Suckfish: a homemade chess engine", author, version)]
struct CmdArgs {
    /// NNUE weight file
    #[arg(long, value_name = "NNUE_PATH", default_value = "nnue.ot")]
    nnue_path: PathBuf,
}

fn main() -> Result<()> {
    let cmd_args = CmdArgs::parse();
    let nnue_runner =
        NnueRunner::new(cmd_args.nnue_path, Device::cuda_if_available())?;
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
                let board = Board::from_fen(fen)?;
                let sr = search_best_move(&board, &mut tt, 4, &nnue_runner);
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
