//! Step 21 (book 5.5) - loading OpenAI's pretrained GPT-2 weights.
//!
//! Training a GPT-2 from scratch costs a fortune; OpenAI published the result. The weights come as
//! a `.safetensors` file, a very plain format: 8 bytes with the header length, a JSON header that
//! says where each tensor lives, then raw f32 numbers. No library needed.
//!
//! Download once (about 500 MB, the file is ignored by git):
//!
//! ```text
//! curl -L -o assets/gpt2/model.safetensors https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
//! ```
//!
//! OpenAI's layout differs from ours in a few ways, which `load_gpt2_weights` fixes:
//! one big `c_attn` matrix holds query, key and value side by side (we keep three); the attention
//! projections carry biases (`qkv_bias = true`); and the output head re-uses the token embedding
//! (weight tying), so we copy `wte` into it. Matrices are stored [in, out], like ours.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::gpt::{CONFIG_124M, Config, GptModel};

pub const GPT2_SMALL_CONFIG: Config = Config {
    context_length: 1024,
    qkv_bias: true,
    drop_rate: 0.0,
    ..CONFIG_124M
};

/// A `.safetensors` file opened for reading tensors by name.
pub struct SafeTensors {
    file: File,
    header: String,
    base: u64,
}

impl SafeTensors {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let n = u64::from_le_bytes(len);
        let mut header = vec![0u8; n as usize];
        file.read_exact(&mut header)?;
        Ok(Self {
            file,
            header: String::from_utf8_lossy(&header).into_owned(),
            base: 8 + n,
        })
    }

    /// The tensor `name` as f32 numbers, and its shape. The header looks like
    /// `"name":{"dtype":"F32","shape":[768,2304],"data_offsets":[0,7077888]}`; we find the entry by name.
    pub fn get(&mut self, name: &str) -> std::io::Result<(Vec<f32>, Vec<usize>)> {
        let bad = |what: &str| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{name}: {what}"))
        };
        let entry = &self.header[self
            .header
            .find(&format!("\"{name}\":"))
            .ok_or_else(|| bad("not in the file"))?..];
        let list = |key: &str| -> Option<Vec<usize>> {
            let at = entry.find(&format!("\"{key}\":["))? + key.len() + 4;
            let inner = &entry[at..at + entry[at..].find(']')?];
            Some(
                inner
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.trim().parse().unwrap())
                    .collect(),
            )
        };
        let shape = list("shape").ok_or_else(|| bad("no shape"))?;
        let offsets = list("data_offsets").ok_or_else(|| bad("no offsets"))?;
        if !entry[..entry.find('}').unwrap()].contains("F32") {
            return Err(bad("only float32 is supported"));
        }
        let mut bytes = vec![0u8; offsets[1] - offsets[0]];
        self.file
            .seek(SeekFrom::Start(self.base + offsets[0] as u64))?;
        self.file.read_exact(&mut bytes)?;
        let values = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok((values, shape))
    }
}

/// Check the shape, then copy.
fn assign(dst: &mut [f32], t: (Vec<f32>, Vec<usize>), expected: usize, what: &str) {
    assert_eq!(
        t.0.len(),
        expected,
        "shape mismatch for {what}: file {:?}",
        t.1
    );
    dst.copy_from_slice(&t.0);
}

/// Build a GPT-2 small model from the weights file.
pub fn load_gpt2_weights(path: &Path) -> std::io::Result<GptModel> {
    let mut st = SafeTensors::open(path)?;
    let cfg = GPT2_SMALL_CONFIG;
    let (d, mut m) = (cfg.emb_dim, GptModel::new(GPT2_SMALL_CONFIG));
    assign(
        &mut m.tok_emb.data,
        st.get("wte.weight")?,
        cfg.vocab_size * d,
        "wte",
    );
    assign(
        &mut m.pos_emb.data,
        st.get("wpe.weight")?,
        cfg.context_length * d,
        "wpe",
    );
    for (i, b) in m.blocks.iter_mut().enumerate() {
        let p = format!("h.{i}.");
        // c_attn is [d, 3d]: query, key and value side by side
        let (c_attn, _) = st.get(&(p.clone() + "attn.c_attn.weight"))?;
        for (which, w) in [&mut b.att.w_query, &mut b.att.w_key, &mut b.att.w_value]
            .into_iter()
            .enumerate()
        {
            for r in 0..d {
                w.data[r * d..(r + 1) * d]
                    .copy_from_slice(&c_attn[r * 3 * d + which * d..r * 3 * d + (which + 1) * d]);
            }
        }
        let (c_bias, _) = st.get(&(p.clone() + "attn.c_attn.bias"))?;
        let qkv = b.att.qkv_bias.as_mut().unwrap();
        for (which, bias) in qkv.iter_mut().enumerate() {
            bias.copy_from_slice(&c_bias[which * d..(which + 1) * d]);
        }
        assign(
            &mut b.att.out_weight.data,
            st.get(&(p.clone() + "attn.c_proj.weight"))?,
            d * d,
            "c_proj",
        );
        assign(
            &mut b.att.out_bias,
            st.get(&(p.clone() + "attn.c_proj.bias"))?,
            d,
            "c_proj bias",
        );
        assign(
            &mut b.ff.expand.w.data,
            st.get(&(p.clone() + "mlp.c_fc.weight"))?,
            d * 4 * d,
            "c_fc",
        );
        assign(
            &mut b.ff.expand.b,
            st.get(&(p.clone() + "mlp.c_fc.bias"))?,
            4 * d,
            "c_fc bias",
        );
        assign(
            &mut b.ff.shrink.w.data,
            st.get(&(p.clone() + "mlp.c_proj.weight"))?,
            4 * d * d,
            "mlp c_proj",
        );
        assign(
            &mut b.ff.shrink.b,
            st.get(&(p.clone() + "mlp.c_proj.bias"))?,
            d,
            "mlp c_proj bias",
        );
        assign(
            &mut b.norm1.scale,
            st.get(&(p.clone() + "ln_1.weight"))?,
            d,
            "ln_1",
        );
        assign(
            &mut b.norm1.shift,
            st.get(&(p.clone() + "ln_1.bias"))?,
            d,
            "ln_1 bias",
        );
        assign(
            &mut b.norm2.scale,
            st.get(&(p.clone() + "ln_2.weight"))?,
            d,
            "ln_2",
        );
        assign(
            &mut b.norm2.shift,
            st.get(&(p + "ln_2.bias"))?,
            d,
            "ln_2 bias",
        );
    }
    assign(&mut m.final_norm.scale, st.get("ln_f.weight")?, d, "ln_f");
    assign(
        &mut m.final_norm.shift,
        st.get("ln_f.bias")?,
        d,
        "ln_f bias",
    );
    let head = st.get("wte.weight")?; // weight tying: the head is the embedding again, [vocab, d] -> [d, vocab]
    for v in 0..cfg.vocab_size {
        for r in 0..d {
            m.out_head.data[r * cfg.vocab_size + v] = head.0[v * d + r];
        }
    }
    Ok(m)
}
