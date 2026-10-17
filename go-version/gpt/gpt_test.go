package gpt

import "testing"

func TestParameterCountMatchesTheBook(t *testing.T) {
	if got := ParameterCount(Config124M); got != 163_009_536 {
		t.Fatalf("got %d, want 163009536", got)
	}
}

func TestBuildingAndArithmeticAgree(t *testing.T) {
	m := NewModel(ConfigTiny)
	if got, want := m.CountParameters(), ParameterCount(ConfigTiny); got != want {
		t.Fatalf("built %d, arithmetic %d", got, want)
	}
}

func TestLayerNormGivesMeanZero(t *testing.T) {
	x := NewLayerNorm(4).Forward(Matrix{Rows: 1, Cols: 4, Data: []float32{1, 2, 3, 10}})
	var sum float32
	for _, v := range x.Data {
		sum += v
	}
	if sum > 1e-5 || sum < -1e-5 {
		t.Fatalf("mean not zero: %v", sum)
	}
}

func TestShortcutKeepsGradientsLarger(t *testing.T) {
	layers := []ExampleLayer{
		{W: [][]float64{{0.5}}, B: []float64{0}},
		{W: [][]float64{{0.5}}, B: []float64{0}},
	}
	plain := (&ExampleDeepNeuralNetwork{Layers: layers}).GradientMeans([]float64{1}, 0)
	with := (&ExampleDeepNeuralNetwork{Layers: layers, UseShortcut: true}).GradientMeans([]float64{1}, 0)
	if with[0] <= plain[0] {
		t.Fatalf("shortcut %v should beat plain %v", with[0], plain[0])
	}
}
