package train

// Step 21 (book 5.5) - loading OpenAI's pretrained GPT-2 weights.
//
// Training a GPT-2 from scratch costs a fortune; OpenAI published the result. The weights come as
// a .safetensors file, a very plain format: 8 bytes with the header length, a JSON header that
// says where each tensor lives, then raw float32 numbers. No library needed.
//
// Download once (about 500 MB, the file is ignored by git):
//
//	curl -L -o assets/gpt2/model.safetensors https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
//
// OpenAI's layout differs from ours in a few ways, which LoadGPT2Weights fixes: one big c_attn
// matrix holds query, key and value side by side (we keep three); the attention projections carry
// biases (QKVBias = true); and the output head re-uses the token embedding (weight tying), so we
// copy wte into it. Matrices are stored [in, out], like ours.

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"llm-from-scratch/gpt"
)

// GPT2Small is the configuration of OpenAI's smallest model.
var GPT2Small = gpt.Config{VocabSize: 50257, ContextLength: 1024, EmbDim: 768, NHeads: 12, NLayers: 12, DropRate: 0, QKVBias: true}

// SafeTensors is a .safetensors file opened for reading tensors by name.
type SafeTensors struct {
	f      *os.File
	header string
	base   int64
}

func OpenSafeTensors(path string) (*SafeTensors, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var n uint64
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	header := make([]byte, n)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, err
	}
	return &SafeTensors{f, string(header), 8 + int64(n)}, nil
}

func (s *SafeTensors) Close() { s.f.Close() }

// list reads `"key":[1,2,3]` from an entry.
func list(entry, key string) []int {
	at := strings.Index(entry, `"`+key+`":[`)
	if at < 0 {
		return nil
	}
	rest := entry[at+len(key)+4:]
	var out []int
	for _, part := range strings.Split(rest[:strings.Index(rest, "]")], ",") {
		if part != "" {
			v, _ := strconv.Atoi(strings.TrimSpace(part))
			out = append(out, v)
		}
	}
	return out
}

// Get returns the tensor name as float32 numbers. The header looks like
// "name":{"dtype":"F32","shape":[768,2304],"data_offsets":[0,7077888]}; we find the entry by name.
func (s *SafeTensors) Get(name string) ([]float32, error) {
	at := strings.Index(s.header, `"`+name+`":`)
	if at < 0 {
		return nil, fmt.Errorf("%s: not in the file", name)
	}
	entry := s.header[at:]
	entry = entry[:strings.Index(entry, "}")]
	if !strings.Contains(entry, "F32") {
		return nil, fmt.Errorf("%s: only float32 is supported", name)
	}
	off := list(entry, "data_offsets")
	buf := make([]byte, off[1]-off[0])
	if _, err := s.f.ReadAt(buf, s.base+int64(off[0])); err != nil {
		return nil, err
	}
	out := make([]float32, len(buf)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*i:]))
	}
	return out, nil
}

// assign checks the size, then copies.
func assign(st *SafeTensors, dst []float32, name string) error {
	src, err := st.Get(name)
	if err != nil {
		return err
	}
	if len(src) != len(dst) {
		return fmt.Errorf("shape mismatch for %s: model %d numbers, file %d", name, len(dst), len(src))
	}
	copy(dst, src)
	return nil
}

// LoadGPT2Weights builds a GPT-2 small model from the weights file.
func LoadGPT2Weights(path string) (*gpt.Model, error) {
	st, err := OpenSafeTensors(path)
	if err != nil {
		return nil, err
	}
	defer st.Close()
	m := gpt.NewModel(GPT2Small)
	d := GPT2Small.EmbDim
	if err := assign(st, m.TokEmb.Data, "wte.weight"); err != nil {
		return nil, err
	}
	if err := assign(st, m.PosEmb.Data, "wpe.weight"); err != nil {
		return nil, err
	}
	for i, b := range m.Blocks {
		p := fmt.Sprintf("h.%d.", i)
		// c_attn is [d, 3d]: query, key and value side by side
		cAttn, err := st.Get(p + "attn.c_attn.weight")
		if err != nil {
			return nil, err
		}
		for which, w := range []gpt.Matrix{b.Att.WQuery, b.Att.WKey, b.Att.WValue} {
			for r := 0; r < d; r++ {
				copy(w.Data[r*d:(r+1)*d], cAttn[r*3*d+which*d:r*3*d+(which+1)*d])
			}
		}
		cBias, err := st.Get(p + "attn.c_attn.bias")
		if err != nil {
			return nil, err
		}
		for which := range b.Att.QKVBias {
			copy(b.Att.QKVBias[which], cBias[which*d:(which+1)*d])
		}
		for _, a := range []struct {
			dst  []float32
			name string
		}{
			{b.Att.OutWeight.Data, "attn.c_proj.weight"}, {b.Att.OutBias, "attn.c_proj.bias"},
			{b.FF.Expand.W.Data, "mlp.c_fc.weight"}, {b.FF.Expand.B, "mlp.c_fc.bias"},
			{b.FF.Shrink.W.Data, "mlp.c_proj.weight"}, {b.FF.Shrink.B, "mlp.c_proj.bias"},
			{b.Norm1.Scale, "ln_1.weight"}, {b.Norm1.Shift, "ln_1.bias"},
			{b.Norm2.Scale, "ln_2.weight"}, {b.Norm2.Shift, "ln_2.bias"},
		} {
			if err := assign(st, a.dst, p+a.name); err != nil {
				return nil, err
			}
		}
	}
	if err := assign(st, m.FinalNorm.Scale, "ln_f.weight"); err != nil {
		return nil, err
	}
	if err := assign(st, m.FinalNorm.Shift, "ln_f.bias"); err != nil {
		return nil, err
	}
	head, err := st.Get("wte.weight") // weight tying: the head is the embedding again, [vocab, d] -> [d, vocab]
	if err != nil {
		return nil, err
	}
	v := GPT2Small.VocabSize
	for tok := 0; tok < v; tok++ {
		for r := 0; r < d; r++ {
			m.OutHead.Data[r*v+tok] = head[tok*d+r]
		}
	}
	return m, nil
}
