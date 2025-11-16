use crate::chess::Board;

use anyhow::{anyhow, bail, Context, Result};
use csv::{ReaderBuilder, StringRecord};
use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tch::nn::init::DEFAULT_KAIMING_UNIFORM;
use tch::nn::OptimizerConfig;
use tch::{nn, Device, Kind, Reduction, Tensor};

const HALF_KA_DIM: i64 = 64 * 12 * 64;
const DEFAULT_FEATURE_DIM: i64 = 256;
const DEFAULT_HIDDEN0_DIM: i64 = 128;
const DEFAULT_HIDDEN1_DIM: i64 = 32;
pub(crate) const CLIPPED_RELU_CAP: f64 = 127.0;
const TARGET_CP_CLIP: f32 = 2000.0;

#[derive(Clone, Copy)]
pub struct NnueConfig {
    pub input_dim: i64,
    pub feature_dim: i64,
    pub hidden0_dim: i64,
    pub hidden1_dim: i64,
}

impl Default for NnueConfig {
    fn default() -> Self {
        Self {
            input_dim: HALF_KA_DIM,
            feature_dim: DEFAULT_FEATURE_DIM,
            hidden0_dim: DEFAULT_HIDDEN0_DIM,
            hidden1_dim: DEFAULT_HIDDEN1_DIM,
        }
    }
}

#[derive(Clone)]
pub struct TrainingOptions {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub max_positions_per_epoch: Option<usize>,
    pub log_interval: usize,
    pub initial_weights: Option<PathBuf>,
    pub seed: Option<i64>,
    pub dedupe_fens: bool,
}

impl Default for TrainingOptions {
    fn default() -> Self {
        Self {
            epochs: 1,
            batch_size: 8192,
            learning_rate: 1e-3,
            max_positions_per_epoch: None,
            log_interval: 50,
            initial_weights: None,
            seed: None,
            dedupe_fens: false,
        }
    }
}

pub struct TrainingSummary {
    pub samples: usize,
    pub batches: usize,
    pub duration: Duration,
}

pub struct NnueRunner {
    _vs: nn::VarStore,
    network: NnueLayers,
    device: Device,
}

impl fmt::Debug for NnueRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NnueRunner")
            .field("device", &self.device)
            .finish()
    }
}

impl NnueRunner {
    /// Initialize NNUE with weights file
    pub fn new(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let mut vs = nn::VarStore::new(device);
        let config = NnueConfig::default();
        let network = NnueLayers::new(&vs.root(), &config);
        vs.load(path.as_ref()).with_context(|| {
            format!("loading NNUE weights from {}", path.as_ref().display())
        })?;

        Ok(Self {
            _vs: vs,
            network,
            device,
        })
    }

    /// Evaluate a board using NNUE
    pub fn eval(&self, board: &Board) -> Result<i32> {
        let indices = board.nnue_active_indices();
        if indices.is_empty() {
            return Err(anyhow!("NNUE state missing active indices"));
        }
        let feat_refs = vec![indices];
        let output = self
            .network
            .forward_sparse(&feat_refs, self.device)
            .squeeze_dim(-1);
        let score = output.squeeze().double_value(&[]).round(); // convert to centipawns
        Ok(score as i32)
    }

    pub fn feature_indices_from_board(&self, board: &Board) -> Result<Vec<i64>> {
        Ok(board.nnue_active_indices().to_vec())
    }
}

pub fn train_from_csv(
    csv_path: impl AsRef<Path>,
    config: NnueConfig,
    options: &TrainingOptions,
    output: &Path,
) -> Result<TrainingSummary> {
    let csv_path = csv_path.as_ref().to_owned();
    let (fen_idx, cp_idx) = resolve_csv_columns(&csv_path)?;
    let dataset_rows = csv_data_row_count(&csv_path)?;
    if dataset_rows == 0 {
        bail!(
            "Training CSV '{}' contains no usable rows.",
            csv_path.display()
        );
    }
    let per_epoch_target = options.max_positions_per_epoch.unwrap_or(dataset_rows);
    let per_epoch_target = per_epoch_target.min(dataset_rows);
    let total_target = if per_epoch_target == 0 {
        None
    } else {
        Some(per_epoch_target * options.epochs)
    };

    if let Some(seed) = options.seed {
        tch::manual_seed(seed);
        if tch::Cuda::is_available() {
            tch::Cuda::manual_seed_all(seed as u64);
        }
    }

    let device = Device::cuda_if_available();
    println!("Loaded NNUE on {:?}.", device);
    let mut vs = nn::VarStore::new(device);
    let network = NnueLayers::new(&vs.root(), &config);
    if let Some(weights) = &options.initial_weights {
        vs.load(weights).with_context(|| {
            format!("loading initial weights from {}", weights.display())
        })?;
    }

    let mut opt = nn::Adam::default().build(&vs, options.learning_rate)?;
    let mut batches = 0usize;
    let mut samples_processed = 0usize;
    let start = Instant::now();
    let batch_size = options.batch_size.max(1);
    for epoch in 0..options.epochs {
        let mut reader = csv_reader(&csv_path)?;
        let mut batch: Vec<Sample> = Vec::with_capacity(batch_size);
        let mut consumed_this_epoch = 0usize;
        let mut seen_fens = if options.dedupe_fens {
            Some(HashSet::with_capacity(per_epoch_target))
        } else {
            None
        };
        for record in reader.records() {
            let record = match record {
                Ok(rec) => rec,
                Err(err) => {
                    eprintln!("Failed to read CSV row: {err}");
                    continue;
                }
            };
            if let Some((sample, hash)) = parse_sample(&record, fen_idx, cp_idx) {
                if let Some(seen) = seen_fens.as_mut() {
                    if !seen.insert(hash) {
                        continue;
                    }
                }
                batch.push(sample);
                consumed_this_epoch += 1;
            }

            let limit_reached = options
                .max_positions_per_epoch
                .is_some_and(|max| consumed_this_epoch >= max);
            if batch.len() == batch_size || (limit_reached && !batch.is_empty()) {
                let loss = train_batch(&network, &batch, device, &mut opt);
                batches += 1;
                samples_processed += batch.len();
                log_progress(&LogProgressArgs {
                    interval: options.log_interval,
                    batch_idx: batches,
                    epoch: epoch + 1,
                    total_epochs: options.epochs,
                    samples_processed,
                    elapsed: start.elapsed(),
                    loss,
                    total_target,
                });
                batch.clear();
            }

            if limit_reached {
                break;
            }
        }

        if !batch.is_empty() {
            let loss = train_batch(&network, &batch, device, &mut opt);
            batches += 1;
            samples_processed += batch.len();
            log_progress(&LogProgressArgs {
                interval: options.log_interval,
                batch_idx: batches,
                epoch: epoch + 1,
                total_epochs: options.epochs,
                samples_processed,
                elapsed: start.elapsed(),
                loss,
                total_target,
            });
        }

        if consumed_this_epoch == 0 {
            break;
        }
    }

    vs.save(output)
        .with_context(|| format!("saving NNUE weights to {}", output.display()))?;

    Ok(TrainingSummary {
        samples: samples_processed,
        batches,
        duration: start.elapsed(),
    })
}

fn csv_reader(path: &Path) -> Result<csv::Reader<std::fs::File>> {
    ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("opening training CSV {}", path.display()))
}

fn resolve_csv_columns(path: &Path) -> Result<(usize, usize)> {
    let mut reader = csv_reader(path)?;
    let headers = reader
        .headers()
        .with_context(|| "reading CSV headers")?
        .clone();
    let fen_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("fen"))
        .ok_or_else(|| anyhow!("CSV missing 'fen' column"))?;
    let cp_idx = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("cp"))
        .ok_or_else(|| anyhow!("CSV missing 'cp' column"))?;
    Ok((fen_idx, cp_idx))
}

fn csv_data_row_count(path: &Path) -> Result<usize> {
    let file = File::open(path).with_context(|| {
        format!("opening training CSV {} for counting", path.display())
    })?;
    let reader = BufReader::new(file);
    let mut lines = 0usize;
    for line in reader.lines() {
        if line.is_ok() {
            lines += 1;
        }
    }
    Ok(lines.saturating_sub(1)) // subtract header
}

fn parse_sample(
    record: &StringRecord,
    fen_idx: usize,
    cp_idx: usize,
) -> Option<(Sample, u64)> {
    let fen = record.get(fen_idx)?.trim();
    if fen.is_empty() {
        return None;
    }
    let cp_str = record.get(cp_idx)?.trim();
    if cp_str.is_empty() {
        return None;
    }
    let cp_value_raw: f32 = match cp_str.parse() {
        Ok(value) => value,
        Err(err) => {
            eprintln!("Skipping row with invalid cp '{cp_str}': {err}");
            return None;
        }
    };
    let cp_value = cp_value_raw.clamp(-TARGET_CP_CLIP, TARGET_CP_CLIP);

    let board = match Board::from_fen(fen) {
        Ok(board) => board,
        Err(err) => {
            eprintln!("Skipping invalid FEN '{fen}': {err}");
            return None;
        }
    };
    let board_hash = board.hash();
    let target = cp_value;
    let features = match board.halfka() {
        Ok(f) => f,
        Err(err) => {
            eprintln!("Skipping board due to HalfKA conversion error: {err}");
            return None;
        }
    };

    Some((Sample { features, target }, board_hash))
}

fn train_batch(
    network: &NnueLayers,
    batch: &[Sample],
    device: Device,
    opt: &mut nn::Optimizer,
) -> f64 {
    let feat_batch = Tensor::stack(
        &batch
            .iter()
            .map(|s| Tensor::from_slice(&s.features))
            .collect::<Vec<_>>(),
        0,
    )
    .to_device(device);
    let targets =
        Tensor::from_slice(&batch.iter().map(|s| s.target).collect::<Vec<_>>())
            .to_device(device);
    let preds = network.forward_dense(&feat_batch).squeeze_dim(-1);
    let loss = preds.mse_loss(&targets, Reduction::Mean);
    opt.backward_step(&loss);
    loss.double_value(&[])
}

struct LogProgressArgs {
    interval: usize,
    batch_idx: usize,
    epoch: usize,
    total_epochs: usize,
    samples_processed: usize,
    elapsed: Duration,
    loss: f64,
    total_target: Option<usize>,
}

fn log_progress(args: &LogProgressArgs) {
    if args.interval == 0 || !args.batch_idx.is_multiple_of(args.interval) {
        return;
    }
    let eta = args
        .total_target
        .and_then(|goal| estimate_eta(args.elapsed, args.samples_processed, goal))
        .unwrap_or_else(|| "?".to_string());
    println!(
        "[epoch {}/{}] batch {} loss {:.4} elapsed {} eta {} samples {}",
        args.epoch,
        args.total_epochs,
        args.batch_idx,
        args.loss,
        format_duration(args.elapsed),
        eta,
        args.samples_processed
    );
}

fn estimate_eta(elapsed: Duration, processed: usize, total: usize) -> Option<String> {
    if processed == 0 || total <= processed {
        return None;
    }
    let elapsed_secs = elapsed.as_secs_f64();
    let rate = processed as f64 / elapsed_secs.max(1.0);
    if rate <= 0.0 {
        return None;
    }
    let remaining = (total - processed) as f64 / rate;
    Some(format_duration(Duration::from_secs_f64(remaining)))
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

struct Sample {
    features: Vec<f32>,
    target: f32,
}

pub(crate) struct NnueLayers {
    pub(crate) view_feat: SparseAccumulator,
    pub(crate) hidden0: nn::Linear,
    pub(crate) hidden1: nn::Linear,
    pub(crate) output: nn::Linear,
    pub(crate) config: NnueConfig,
}

impl NnueLayers {
    pub(crate) fn new(path: &nn::Path, config: &NnueConfig) -> Self {
        let linear_cfg = nn::LinearConfig {
            bias: true,
            ..Default::default()
        };
        let view_feat = SparseAccumulator::new(
            &(path / "view_feat"),
            config.input_dim,
            config.feature_dim,
        );
        let hidden0 = nn::linear(
            path / "hidden0",
            config.feature_dim,
            config.hidden0_dim,
            linear_cfg,
        );
        let hidden1 = nn::linear(
            path / "hidden1",
            config.hidden0_dim,
            config.hidden1_dim,
            linear_cfg,
        );
        let output = nn::linear(path / "output", config.hidden1_dim, 1, linear_cfg);

        Self {
            view_feat,
            hidden0,
            hidden1,
            output,
            config: *config,
        }
    }

    fn forward_dense(&self, features: &Tensor) -> Tensor {
        let mut x = clipped_relu(self.view_feat.forward_dense(features));
        x = clipped_relu(x.apply(&self.hidden0));
        x = clipped_relu(x.apply(&self.hidden1));
        x.apply(&self.output)
    }

    fn forward_sparse(&self, feat_indices: &[&[i64]], device: Device) -> Tensor {
        let feat_activations: Vec<Tensor> = feat_indices
            .iter()
            .map(|idx| {
                let idx_tensor = indices_tensor(idx, device);
                self.view_feat.forward(&idx_tensor)
            })
            .collect();
        let mut x = clipped_relu(Tensor::stack(&feat_activations, 0));
        x = clipped_relu(x.apply(&self.hidden0));
        x = clipped_relu(x.apply(&self.hidden1));
        x.apply(&self.output)
    }

    #[allow(dead_code)]
    fn input_dim(&self) -> i64 {
        self.config.input_dim
    }
}

pub(crate) struct SparseAccumulator {
    pub(crate) weights: Tensor,
    pub(crate) bias: Tensor,
}

impl SparseAccumulator {
    fn new(path: &nn::Path, input_dim: i64, output_dim: i64) -> Self {
        let weights =
            path.var("weight", &[input_dim, output_dim], DEFAULT_KAIMING_UNIFORM);
        let bias = path.var("bias", &[output_dim], nn::Init::Const(0.0));
        Self { weights, bias }
    }

    fn forward(&self, indices: &Tensor) -> Tensor {
        if indices.numel() == 0 {
            return self.bias.shallow_clone();
        }
        let sum_dim = [0];
        let gathered = self.weights.index_select(0, indices);
        gathered.sum_dim_intlist(&sum_dim[..], false, Kind::Float) + &self.bias
    }

    fn forward_dense(&self, inputs: &Tensor) -> Tensor {
        inputs.matmul(&self.weights) + &self.bias
    }
}

fn indices_tensor(indices: &[i64], device: Device) -> Tensor {
    if indices.is_empty() {
        Tensor::zeros([0], (Kind::Int64, device))
    } else {
        Tensor::from_slice(indices).to_device(device)
    }
}

fn clipped_relu(tensor: Tensor) -> Tensor {
    tensor.relu().clamp_max(CLIPPED_RELU_CAP)
}
