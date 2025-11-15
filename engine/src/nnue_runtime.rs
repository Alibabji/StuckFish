use crate::chess::Board;
use crate::nnue::{self, NnueConfig, NnueLayers};
use anyhow::{Context, Result, anyhow};
use std::convert::TryInto;
use std::path::Path;
use tch::{Device, Tensor, nn};

pub struct NnueRuntime {
    view: SparseAccumulatorRuntime,
    hidden0: LinearLayer,
    hidden1: LinearLayer,
    output: LinearLayer,
}

impl NnueRuntime {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let mut vs = nn::VarStore::new(Device::Cpu);
        let layers = NnueLayers::new(&vs.root(), &NnueConfig::default());
        vs.load(path.as_ref()).with_context(|| {
            format!("loading NNUE weights from {}", path.as_ref().display())
        })?;
        Self::from_layers(&layers)
    }

    fn from_layers(layers: &NnueLayers) -> Result<Self> {
        let view = SparseAccumulatorRuntime::from_tensor(
            &layers.view_feat.weights,
            &layers.view_feat.bias,
        )?;
        let hidden0 = LinearLayer::from_linear(&layers.hidden0)?;
        let hidden1 = LinearLayer::from_linear(&layers.hidden1)?;
        let output = LinearLayer::from_linear(&layers.output)?;
        Ok(Self {
            view,
            hidden0,
            hidden1,
            output,
        })
    }

    pub fn eval(&self, board: &Board) -> Result<i32> {
        let indices = board.nnue_active_indices();
        if indices.is_empty() {
            return Err(anyhow!("NNUE accumulator is empty"));
        }
        let mut x = self.view.forward(indices);
        clipped_relu(&mut x);
        x = self.hidden0.forward(&x);
        clipped_relu(&mut x);
        x = self.hidden1.forward(&x);
        clipped_relu(&mut x);
        let out = self.output.forward(&x);
        let score = out
            .first()
            .copied()
            .ok_or_else(|| anyhow!("NNUE output missing value"))?;
        Ok(score.round() as i32)
    }
}

fn clipped_relu(values: &mut [f32]) {
    for v in values {
        if *v < 0.0 {
            *v = 0.0;
        } else if *v > nnue::CLIPPED_RELU_CAP as f32 {
            *v = nnue::CLIPPED_RELU_CAP as f32;
        }
    }
}

struct SparseAccumulatorRuntime {
    weights: Vec<f32>,
    bias: Vec<f32>,
    output_dim: usize,
}

impl SparseAccumulatorRuntime {
    fn from_tensor(weights: &Tensor, bias: &Tensor) -> Result<Self> {
        let weights = tensor_to_vec(weights)?;
        let bias = tensor_to_vec(bias)?;
        let output_dim = bias.len();
        Ok(Self {
            weights,
            bias,
            output_dim,
        })
    }

    fn forward(&self, indices: &[i64]) -> Vec<f32> {
        let mut acc = self.bias.clone();
        if indices.is_empty() {
            return acc;
        }
        for &idx in indices {
            let offset = idx as usize * self.output_dim;
            let row = &self.weights[offset..offset + self.output_dim];
            for (dest, w) in acc.iter_mut().zip(row.iter()) {
                *dest += *w;
            }
        }
        acc
    }
}

struct LinearLayer {
    weights: Vec<f32>,
    bias: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl LinearLayer {
    fn from_linear(linear: &nn::Linear) -> Result<Self> {
        let ws = tensor_to_vec(&linear.ws)?;
        let in_dim = linear.ws.size()[1] as usize;
        let out_dim = linear.ws.size()[0] as usize;
        let bias = if let Some(b) = &linear.bs {
            tensor_to_vec(b)?
        } else {
            vec![0.0; out_dim]
        };
        if ws.len() != in_dim * out_dim {
            return Err(anyhow!("invalid NNUE linear weight size"));
        }
        Ok(Self {
            weights: ws,
            bias,
            in_dim,
            out_dim,
        })
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.in_dim);
        let mut out = vec![0.0; self.out_dim];
        for o in 0..self.out_dim {
            let mut sum = self.bias[o];
            let row = &self.weights[o * self.in_dim..(o + 1) * self.in_dim];
            for i in 0..self.in_dim {
                sum += row[i] * input[i];
            }
            out[o] = sum;
        }
        out
    }
}

fn tensor_to_vec(tensor: &Tensor) -> Result<Vec<f32>> {
    let cpu = tensor.to_device(Device::Cpu);
    let flattened = cpu.flatten(0, -1);
    let vec: Vec<f32> = flattened
        .try_into()
        .map_err(|e| anyhow!("failed to read tensor data: {e}"))?;
    Ok(vec)
}
