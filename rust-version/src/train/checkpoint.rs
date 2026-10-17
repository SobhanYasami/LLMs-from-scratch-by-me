//! Step 20 (book 5.4) - saving and loading a model (and the optimizer).
//!
//! Training is expensive, so save the weights and reload them later. The book uses
//! `torch.save(model.state_dict())`. To keep the file readable by every language in this repo, we
//! write one plain format:
//!
//! ```text
//! "LLMCKPT1" | 7 x u32: vocab, context, emb, heads, layers, qkv_bias, has_optimizer
//! then raw little-endian f32 numbers, tensor after tensor, in the order of `GptModel::tensors`
//! (the weights), then the AdamW state if present: step (as f32), then exp_avg and exp_avg_sq of
//! every tensor.
//! ```
//!
//! A checkpoint written by Rust can be loaded by Python, Go, C or Zig, and the other way round.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use super::adamw::AdamW;
use crate::gpt::{Config, GptModel};

const MAGIC: &[u8; 8] = b"LLMCKPT1";

fn write_f32s(w: &mut impl Write, values: &[f32]) -> std::io::Result<()> {
    for v in values {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_f32s(r: &mut impl Read, out: &mut [f32]) -> std::io::Result<()> {
    let mut buf = [0u8; 4];
    for v in out.iter_mut() {
        r.read_exact(&mut buf)?;
        *v = f32::from_le_bytes(buf);
    }
    Ok(())
}

pub fn save_checkpoint(
    path: &Path,
    model: &GptModel,
    optimizer: Option<&AdamW>,
) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    let c = &model.cfg;
    w.write_all(MAGIC)?;
    for v in [
        c.vocab_size,
        c.context_length,
        c.emb_dim,
        c.n_heads,
        c.n_layers,
        c.qkv_bias as usize,
        optimizer.is_some() as usize,
    ] {
        w.write_all(&(v as u32).to_le_bytes())?;
    }
    for t in model.tensors() {
        write_f32s(&mut w, t)?;
    }
    if let Some(opt) = optimizer {
        write_f32s(&mut w, &[opt.step as f32])?;
        for t in opt.exp_avg.tensors() {
            write_f32s(&mut w, t)?;
        }
        for t in opt.exp_avg_sq.tensors() {
            write_f32s(&mut w, t)?;
        }
    }
    w.flush()
}

/// Read a checkpoint. Returns the model, and the optimizer state if the file has it
/// (`lr` and `weight_decay` are settings, not saved).
pub fn load_checkpoint(
    path: &Path,
    lr: f64,
    weight_decay: f64,
) -> std::io::Result<(GptModel, Option<AdamW>)> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a checkpoint of this repo",
        ));
    }
    let mut h = [0u32; 7];
    let mut b4 = [0u8; 4];
    for v in h.iter_mut() {
        r.read_exact(&mut b4)?;
        *v = u32::from_le_bytes(b4);
    }
    let cfg = Config {
        vocab_size: h[0] as usize,
        context_length: h[1] as usize,
        emb_dim: h[2] as usize,
        n_heads: h[3] as usize,
        n_layers: h[4] as usize,
        drop_rate: 0.0,
        qkv_bias: h[5] != 0,
    };
    let mut model = GptModel::new(cfg);
    for t in model.tensors_mut() {
        read_f32s(&mut r, t)?;
    }
    let optimizer = if h[6] != 0 {
        let mut opt = AdamW::new(&model, lr, weight_decay);
        let mut step = [0f32; 1];
        read_f32s(&mut r, &mut step)?;
        opt.step = step[0] as u32;
        for t in opt.exp_avg.tensors_mut() {
            read_f32s(&mut r, t)?;
        }
        for t in opt.exp_avg_sq.tensors_mut() {
            read_f32s(&mut r, t)?;
        }
        Some(opt)
    } else {
        None
    };
    Ok((model, optimizer))
}
