package gpt

// Step 13 (book 4.4) - shortcut connections, and why they matter.
//
// A shortcut (residual) connection adds a layer's input to its output:
//
//	x = x + layer(x)        instead of        x = layer(x)
//
// During training the gradient of the loss must travel backwards through every
// layer. Without shortcuts it shrinks a little at each layer (the vanishing
// gradient problem), so the first layers barely learn. With shortcuts the
// gradient also has a short path straight back, so it stays healthy.
//
// Python gets the gradients from torch's autograd. Here there is no autograd,
// so this file does the backward pass by hand, in float64: the chain rule,
// layer by layer, from the loss back to the first layer. It reproduces the
// book's numbers.

import "math"

// ExampleLayer is Linear + GELU with W stored [out][in] like nn.Linear.
type ExampleLayer struct {
	W [][]float64
	B []float64
}

type ExampleDeepNeuralNetwork struct {
	Layers      []ExampleLayer
	UseShortcut bool
}

func gelu64(v float64) float64 {
	return 0.5 * v * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(v+0.044715*v*v*v)))
}

// geluGrad is the derivative of the tanh-approximated GELU.
func geluGrad(v float64) float64 {
	c := math.Sqrt(2 / math.Pi)
	t := math.Tanh(c * (v + 0.044715*v*v*v))
	return 0.5*(1+t) + 0.5*v*(1-t*t)*c*(1+3*0.044715*v*v)
}

// GradientMeans runs one forward and backward pass (loss = (output - target)^2)
// and returns the mean absolute gradient of each layer's weight matrix, first layer first.
func (n *ExampleDeepNeuralNetwork) GradientMeans(x []float64, target float64) []float64 {
	acts := [][]float64{x} // input of every layer
	pre := [][]float64{}   // Linear output (before GELU) of every layer
	shortcut := []bool{}
	for _, l := range n.Layers {
		in := acts[len(acts)-1]
		z := make([]float64, len(l.W))
		for i, row := range l.W {
			z[i] = l.B[i]
			for j, w := range row {
				z[i] += w * in[j]
			}
		}
		out := make([]float64, len(z))
		for i, v := range z {
			out[i] = gelu64(v)
		}
		// A shortcut needs matching shapes, so the last (3 -> 1) layer has none.
		sc := n.UseShortcut && len(out) == len(in)
		if sc {
			for i := range out {
				out[i] += in[i]
			}
		}
		pre, acts, shortcut = append(pre, z), append(acts, out), append(shortcut, sc)
	}

	grad := []float64{2 * (acts[len(acts)-1][0] - target)} // d loss / d output
	means := make([]float64, len(n.Layers))
	for li := len(n.Layers) - 1; li >= 0; li-- {
		l, in := n.Layers[li], acts[li]
		dz := make([]float64, len(l.W)) // gradient at the Linear output
		for i := range dz {
			dz[i] = grad[i] * geluGrad(pre[li][i])
		}
		var sum float64
		for i, row := range l.W {
			for j := range row {
				sum += math.Abs(dz[i] * in[j]) // d loss / d W[i][j]
			}
		}
		means[li] = sum / float64(len(l.W)*len(l.W[0]))
		next := make([]float64, len(in)) // gradient handed to the layer below
		for j := range next {
			for i, row := range l.W {
				next[j] += row[j] * dz[i]
			}
			if shortcut[li] {
				next[j] += grad[j] // the shortcut: gradient also flows straight through
			}
		}
		grad = next
	}
	return means
}
