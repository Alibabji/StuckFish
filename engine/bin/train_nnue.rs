use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use suckfish::nnue::{self, NnueConfig, TrainingOptions};

#[derive(Parser, Debug)]
#[command(about = "Train the NNUE network", author, version)]
struct TrainArgs {
    /// Random seed for reproducible runs
    #[arg(long, value_name = "SEED")]
    seed: Option<i64>,

    /// CSV file with columns: fen, cp[, depth][, mate]
    #[arg(long, value_name = "FILE", required = true)]
    csv: PathBuf,

    /// Where to write the trained weights (.ot file)
    #[arg(long, value_name = "PATH", default_value = "nnue/network.ot")]
    output: PathBuf,

    /// Number of passes over the dataset
    #[arg(long, default_value_t = 1)]
    epochs: usize,

    /// Mini-batch size
    #[arg(long, default_value_t = 8192)]
    batch_size: usize,

    /// Optimizer learning rate
    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f64,

    /// Upper bound of positions to consume per epoch
    #[arg(long)]
    max_positions: Option<usize>,

    /// Print a training update every N mini-batches
    #[arg(long, default_value_t = 50)]
    log_interval: usize,

    /// Existing weights file to initialize from (resume training)
    #[arg(long, value_name = "FILE")]
    init_weights: Option<PathBuf>,

    /// Ignore duplicate FENs within each epoch (costs extra RAM)
    #[arg(long)]
    dedupe_fens: bool,
}

fn main() -> Result<()> {
    let args = TrainArgs::parse();
    let mut options = TrainingOptions::default();
    options.epochs = args.epochs;
    options.batch_size = args.batch_size;
    options.learning_rate = args.learning_rate;
    options.max_positions_per_epoch =
        args.max_positions.or(options.max_positions_per_epoch);
    options.log_interval = args.log_interval;
    options.initial_weights = args.init_weights.clone();
    options.seed = args.seed;
    options.dedupe_fens = args.dedupe_fens;

    fs::create_dir_all(
        args.output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )
    .with_context(|| {
        format!("creating output directory for {}", args.output.display())
    })?;

    let summary = nnue::train_from_csv(
        args.csv,
        NnueConfig::default(),
        &options,
        &args.output,
    )?;
    println!(
        "Training complete: {} samples across {} batches in {:.2?}. Weights saved to {}.",
        summary.samples,
        summary.batches,
        summary.duration,
        args.output.display()
    );
    Ok(())
}
