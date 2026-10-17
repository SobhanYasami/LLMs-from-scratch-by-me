package attention

import "math"

// Tiny seeded random generator (splitmix64), so every language gives the same
// numbers. Used for random weights and for dropout masks.
type Rng struct{ state uint64 }

func NewRng(seed uint64) *Rng { return &Rng{seed} }

func (r *Rng) nextU64() uint64 {
	r.state += 0x9E3779B97F4A7C15
	z := r.state
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

// NextF64 is uniform in [0, 1).
func (r *Rng) NextF64() float64 { return float64(r.nextU64()>>11) / 9007199254740992.0 }

// Normal is a Normal(0, 1) draw (Box-Muller). It uses two uniforms and keeps
// one value, so every language walks the same random stream.
func (r *Rng) Normal() float32 {
	u1 := 1 - r.NextF64() // avoid log(0)
	u2 := r.NextF64()
	return float32(math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2))
}

// Randn returns a rows x cols matrix of Normal(0, std) values.
func (r *Rng) Randn(rows, cols int, std float32) Matrix {
	m := NewMatrix(rows, cols)
	for i := range m.Data {
		m.Data[i] = r.Normal() * std
	}
	return m
}

// Rand returns a rows x cols matrix, uniform in [0, 1), like torch.rand.
func (r *Rng) Rand(rows, cols int) Matrix {
	m := NewMatrix(rows, cols)
	for i := range m.Data {
		m.Data[i] = float32(r.NextF64())
	}
	return m
}

// Dropout zeroes each element with probability p and scales the rest by 1/(1-p).
// The scaling keeps the average unchanged, so nothing needs rescaling at
// inference time (when dropout is simply switched off).
func Dropout(m Matrix, p float32, r *Rng) Matrix {
	out := NewMatrix(m.Rows, m.Cols)
	for i, v := range m.Data {
		if r.NextF64() >= float64(p) {
			out.Data[i] = v / (1 - p)
		}
	}
	return out
}
