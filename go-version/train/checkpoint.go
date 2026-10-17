package train

// Step 20 (book 5.4) - saving and loading a model (and the optimizer).
//
// Training is expensive, so save the weights and reload them later. The book uses
// torch.save(model.state_dict()). To keep the file readable by every language in this repo, we
// write one plain format:
//
//	"LLMCKPT1" | 7 x uint32: vocab, context, emb, heads, layers, qkv_bias, has_optimizer
//	then raw little-endian float32 numbers, tensor after tensor, in the order of Model.Tensors
//	(the weights), then the AdamW state if present: step (as float32), then exp_avg and
//	exp_avg_sq of every tensor.
//
// A checkpoint written by Go can be loaded by Python, Rust, C or Zig, and the other way round.

import (
	"bufio"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"

	"llm-from-scratch/gpt"
)

const magic = "LLMCKPT1"

func writeFloats(w io.Writer, values []float32) error {
	buf := make([]byte, 4*len(values))
	for i, v := range values {
		binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
	}
	_, err := w.Write(buf)
	return err
}

func readFloats(r io.Reader, out []float32) error {
	buf := make([]byte, 4*len(out))
	if _, err := io.ReadFull(r, buf); err != nil {
		return err
	}
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
	}
	return nil
}

// SaveCheckpoint writes the model and, if opt is not nil, the optimizer state.
func SaveCheckpoint(path string, m *gpt.Model, opt *AdamW) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	c := m.Cfg
	w.WriteString(magic)
	hasOpt, qkv := 0, 0
	if opt != nil {
		hasOpt = 1
	}
	if c.QKVBias {
		qkv = 1
	}
	for _, v := range []int{c.VocabSize, c.ContextLength, c.EmbDim, c.NHeads, c.NLayers, qkv, hasOpt} {
		binary.Write(w, binary.LittleEndian, uint32(v))
	}
	for _, t := range m.Tensors() {
		if err := writeFloats(w, t); err != nil {
			return err
		}
	}
	if opt != nil {
		writeFloats(w, []float32{float32(opt.Step)})
		for _, set := range []*gpt.Model{opt.ExpAvg, opt.ExpAvgSq} {
			for _, t := range set.Tensors() {
				if err := writeFloats(w, t); err != nil {
					return err
				}
			}
		}
	}
	return w.Flush()
}

// LoadCheckpoint reads a checkpoint. It returns the model, and the optimizer state if the file has
// it (lr and weightDecay are settings, not saved).
func LoadCheckpoint(path string, lr, weightDecay float64) (*gpt.Model, *AdamW, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	head := make([]byte, 8)
	if _, err := io.ReadFull(r, head); err != nil || string(head) != magic {
		return nil, nil, errors.New("not a checkpoint of this repo")
	}
	var h [7]uint32
	if err := binary.Read(r, binary.LittleEndian, &h); err != nil {
		return nil, nil, err
	}
	m := gpt.NewModel(gpt.Config{VocabSize: int(h[0]), ContextLength: int(h[1]), EmbDim: int(h[2]), NHeads: int(h[3]), NLayers: int(h[4]), QKVBias: h[5] != 0})
	for _, t := range m.Tensors() {
		if err := readFloats(r, t); err != nil {
			return nil, nil, err
		}
	}
	if h[6] == 0 {
		return m, nil, nil
	}
	opt := NewAdamW(m, lr, weightDecay)
	step := make([]float32, 1)
	if err := readFloats(r, step); err != nil {
		return nil, nil, err
	}
	opt.Step = int(step[0])
	for _, set := range []*gpt.Model{opt.ExpAvg, opt.ExpAvgSq} {
		for _, t := range set.Tensors() {
			if err := readFloats(r, t); err != nil {
				return nil, nil, err
			}
		}
	}
	return m, opt, nil
}
