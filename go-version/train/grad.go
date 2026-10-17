package train

// Step 18 (book 5.2) - the backward pass, by hand.
//
// Training needs to know, for EVERY weight, "if I make this weight a little bigger, does the loss
// go up or down, and how fast?" That number is the weight's gradient. torch's autograd finds all
// of them automatically; here we do the same with the chain rule:
//
//  1. run the forward pass and REMEMBER what each layer saw and produced (the "cache");
//  2. start at the loss and walk the layers backwards, turning "how the loss depends on this
//     layer's output" into "how it depends on its weights and on its input".
//
// Every layer has a small backward rule:
//
//	y = x W + b       dW = x^T dy     db = sum of dy over tokens     dx = dy W^T
//	y = gelu(x)       dx = dy * gelu'(x)
//	y = layer_norm(x) see lnBackward
//	y = x + f(x)      dx = dy + (what flows back through f)        <- the shortcut, chapter 4.4
//	attention         softmax backward: dS = P * (dP - sum(dP * P) per row)
//
// The gradients are stored in a gpt.Model of the same shape (a handy trick: the optimizer and the
// checkpoint code then treat weights, gradients and optimizer state alike).

import (
	"math"

	"llm-from-scratch/attention"
	"llm-from-scratch/gpt"
	"llm-from-scratch/tokenization"
)

type Matrix = attention.Matrix

const lnEps = 1e-5

func attnScale(headDim int) float32 { return 1 / float32(math.Sqrt(float64(headDim))) }

func addInto(dst, src Matrix) {
	for i, s := range src.Data {
		dst.Data[i] += s
	}
}

func addBias(m Matrix, bias []float32) {
	for i := 0; i < m.Rows; i++ {
		row := m.Row(i)
		for j := range row {
			row[j] += bias[j]
		}
	}
}

// geluGrad is d gelu(x) / dx for the tanh approximation.
func geluGrad(x float32) float32 {
	v := float64(x)
	c := math.Sqrt(2 / math.Pi)
	t := math.Tanh(c * (v + 0.044715*v*v*v))
	return float32(0.5*(1+t) + 0.5*v*(1-t*t)*c*(1+3*0.044715*v*v))
}

// lnCache is what layer norm remembers for the backward pass.
type lnCache struct {
	xhat Matrix    // the normalized values, before scale and shift
	inv  []float32 // 1 / sqrt(variance + eps) per row
}

func lnForward(ln *gpt.LayerNorm, x Matrix) (Matrix, lnCache) {
	out, xhat := attention.NewMatrix(x.Rows, x.Cols), attention.NewMatrix(x.Rows, x.Cols)
	inv := make([]float32, x.Rows)
	n := float32(x.Cols)
	for i := 0; i < x.Rows; i++ {
		row := x.Row(i)
		var mean float32
		for _, v := range row {
			mean += v
		}
		mean /= n
		var variance float32
		for _, v := range row {
			variance += (v - mean) * (v - mean)
		}
		variance /= n
		inv[i] = float32(1 / math.Sqrt(float64(variance+lnEps)))
		for j, v := range row {
			h := (v - mean) * inv[i]
			xhat.Set(i, j, h)
			out.Set(i, j, ln.Scale[j]*h+ln.Shift[j])
		}
	}
	return out, lnCache{xhat, inv}
}

// lnBackward: with dxhat = dy * scale,
// dx = inv * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)); dscale += dy * xhat; dshift += dy.
func lnBackward(ln *gpt.LayerNorm, c lnCache, dy Matrix, dscale, dshift []float32) Matrix {
	dx := attention.NewMatrix(dy.Rows, dy.Cols)
	n := float32(dy.Cols)
	for i := 0; i < dy.Rows; i++ {
		var s1, s2 float32
		for j := 0; j < dy.Cols; j++ {
			dxhat := dy.At(i, j) * ln.Scale[j]
			s1 += dxhat
			s2 += dxhat * c.xhat.At(i, j)
			dscale[j] += dy.At(i, j) * c.xhat.At(i, j)
			dshift[j] += dy.At(i, j)
		}
		for j := 0; j < dy.Cols; j++ {
			dxhat := dy.At(i, j) * ln.Scale[j]
			dx.Set(i, j, c.inv[i]*(dxhat-s1/n-c.xhat.At(i, j)*(s2/n)))
		}
	}
	return dx
}

// linearBackward: y = x W (+ b). Returns dx and adds to dW and db (db may be nil).
func linearBackward(x, w, dy Matrix, dw Matrix, db []float32) Matrix {
	addInto(dw, attention.MatMul(attention.Transpose(x), dy))
	if db != nil {
		for i := 0; i < dy.Rows; i++ {
			for j, v := range dy.Row(i) {
				db[j] += v
			}
		}
	}
	return attention.MatMul(dy, attention.Transpose(w))
}

// blockCache is everything one transformer block saw in the forward pass.
type blockCache struct {
	ln1       lnCache
	n1        Matrix
	q, k, v   Matrix
	probs     []Matrix // attention weights of each head [T, T]
	ctx       Matrix
	ln2       lnCache
	n2, h1, g Matrix
}

// forwardCached is the forward pass of the whole model for one sequence, keeping the cache. Returns the logits.
func forwardCached(m *gpt.Model, ids []int) (Matrix, []blockCache, Matrix, lnCache) {
	x := m.Embed(ids)
	var caches []blockCache
	for _, b := range m.Blocks {
		a := b.Att
		xIn := x
		n1, ln1 := lnForward(b.Norm1, xIn)
		q, k, v := attention.MatMul(n1, a.WQuery), attention.MatMul(n1, a.WKey), attention.MatMul(n1, a.WValue)
		ctx := attention.NewMatrix(x.Rows, a.NumHeads*a.HeadDim)
		var probs []Matrix
		for h := 0; h < a.NumHeads; h++ {
			lo, hi := h*a.HeadDim, (h+1)*a.HeadDim
			qh, kh, vh := attention.SliceCols(q, lo, hi), attention.SliceCols(k, lo, hi), attention.SliceCols(v, lo, hi)
			scores := attention.MaskFuture(attention.MatMul(qh, attention.Transpose(kh)))
			weights := attention.SoftmaxRows(attention.Scale(scores, attnScale(a.HeadDim)))
			head := attention.MatMul(weights, vh)
			for i := 0; i < head.Rows; i++ {
				copy(ctx.Row(i)[lo:hi], head.Row(i))
			}
			probs = append(probs, weights)
		}
		xMid := attention.MatMul(ctx, a.OutWeight)
		addBias(xMid, a.OutBias)
		addInto(xMid, xIn)
		n2, ln2 := lnForward(b.Norm2, xMid)
		h1 := attention.MatMul(n2, b.FF.Expand.W)
		addBias(h1, b.FF.Expand.B)
		g := attention.NewMatrix(h1.Rows, h1.Cols)
		for i, v := range h1.Data {
			g.Data[i] = gpt.GELU(v)
		}
		f := attention.MatMul(g, b.FF.Shrink.W)
		addBias(f, b.FF.Shrink.B)
		addInto(f, xMid)
		x = f
		caches = append(caches, blockCache{ln1, n1, q, k, v, probs, ctx, ln2, n2, h1, g})
	}
	nf, lnf := lnForward(m.FinalNorm, x)
	return attention.MatMul(nf, m.OutHead), caches, nf, lnf
}

// LossAndGrads returns the mean cross entropy of the batch and ADDS the gradient of that mean for
// every weight into grads (zero it first with ZeroGrads).
func LossAndGrads(m, grads *gpt.Model, batch tokenization.Batch) float32 {
	for _, b := range m.Blocks {
		if b.Att.QKVBias != nil {
			panic("training without the attention biases only")
		}
	}
	nTokens := 0
	for _, t := range batch.Targets {
		nTokens += len(t)
	}
	var total float32
	for si, ids := range batch.Inputs {
		targets := batch.Targets[si]
		logits, caches, nf, lnf := forwardCached(m, ids)

		// loss and its gradient: d loss / d logits = (softmax - one_hot) / N
		dlogits := attention.NewMatrix(logits.Rows, logits.Cols)
		for i, t := range targets {
			row := logits.Row(i)
			total += LogSumExp(row) - row[t]
			max := float32(math.Inf(-1))
			for _, v := range row {
				if v > max {
					max = v
				}
			}
			var sum float32
			for _, v := range row {
				sum += float32(math.Exp(float64(v - max)))
			}
			for j, v := range row {
				p := float32(math.Exp(float64(v-max))) / sum
				var oneHot float32
				if j == t {
					oneHot = 1
				}
				dlogits.Set(i, j, (p-oneHot)/float32(nTokens))
			}
		}

		// output head and final layer norm
		dnf := linearBackward(nf, m.OutHead, dlogits, grads.OutHead, nil)
		dx := lnBackward(m.FinalNorm, lnf, dnf, grads.FinalNorm.Scale, grads.FinalNorm.Shift)

		// blocks, last to first
		for bi := len(m.Blocks) - 1; bi >= 0; bi-- {
			b, c, gb := m.Blocks[bi], caches[bi], grads.Blocks[bi]
			a := b.Att
			// feed forward: x_out = x_mid + ff(ln2(x_mid))
			dh1 := linearBackward(c.g, b.FF.Shrink.W, dx, gb.FF.Shrink.W, gb.FF.Shrink.B)
			for i, h := range c.h1.Data {
				dh1.Data[i] *= geluGrad(h)
			}
			dn2 := linearBackward(c.n2, b.FF.Expand.W, dh1, gb.FF.Expand.W, gb.FF.Expand.B)
			dxMid := lnBackward(b.Norm2, c.ln2, dn2, gb.Norm2.Scale, gb.Norm2.Shift)
			addInto(dxMid, dx) // the shortcut
			// attention: x_mid = x_in + att(ln1(x_in))
			dctx := linearBackward(c.ctx, a.OutWeight, dxMid, gb.Att.OutWeight, gb.Att.OutBias)
			dq, dk, dv := attention.NewMatrix(c.q.Rows, c.q.Cols), attention.NewMatrix(c.q.Rows, c.q.Cols), attention.NewMatrix(c.q.Rows, c.q.Cols)
			for h := 0; h < a.NumHeads; h++ {
				lo, hi := h*a.HeadDim, (h+1)*a.HeadDim
				qh, kh, vh := attention.SliceCols(c.q, lo, hi), attention.SliceCols(c.k, lo, hi), attention.SliceCols(c.v, lo, hi)
				p := c.probs[h]
				dctxH := attention.SliceCols(dctx, lo, hi)
				dp := attention.MatMul(dctxH, attention.Transpose(vh))
				dvh := attention.MatMul(attention.Transpose(p), dctxH)
				// softmax backward, then the 1/sqrt(d) scale (masked entries have p = 0, so no gradient)
				ds := attention.NewMatrix(p.Rows, p.Cols)
				for i := 0; i < p.Rows; i++ {
					var dot float32
					for j := 0; j < p.Cols; j++ {
						dot += dp.At(i, j) * p.At(i, j)
					}
					for j := 0; j < p.Cols; j++ {
						ds.Set(i, j, p.At(i, j)*(dp.At(i, j)-dot))
					}
				}
				ds = attention.Scale(ds, attnScale(a.HeadDim))
				dqh := attention.MatMul(ds, kh)
				dkh := attention.MatMul(attention.Transpose(ds), qh)
				for i := 0; i < dqh.Rows; i++ {
					for j := 0; j < a.HeadDim; j++ {
						dq.Set(i, lo+j, dqh.At(i, j))
						dk.Set(i, lo+j, dkh.At(i, j))
						dv.Set(i, lo+j, dvh.At(i, j))
					}
				}
			}
			dn1 := linearBackward(c.n1, a.WQuery, dq, gb.Att.WQuery, nil)
			addInto(dn1, linearBackward(c.n1, a.WKey, dk, gb.Att.WKey, nil))
			addInto(dn1, linearBackward(c.n1, a.WValue, dv, gb.Att.WValue, nil))
			dx = lnBackward(b.Norm1, c.ln1, dn1, gb.Norm1.Scale, gb.Norm1.Shift)
			addInto(dx, dxMid) // the shortcut
		}

		// embeddings: every token's row of dx goes back to its row of the token table
		for t, id := range ids {
			for j := 0; j < dx.Cols; j++ {
				grads.TokEmb.Data[id*dx.Cols+j] += dx.At(t, j)
				grads.PosEmb.Data[t*dx.Cols+j] += dx.At(t, j)
			}
		}
	}
	return total / float32(nTokens)
}

// ZeroGrads sets every gradient to 0.
func ZeroGrads(grads *gpt.Model) {
	for _, t := range grads.Tensors() {
		clear(t)
	}
}
