package gonn

import (
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestBackPropagation(t *testing.T) {
	const (
		desiredAnswer = 0
		actualAnswer  = 0.03477387984894708
		seed          = 0
	)

	rand.Seed(seed)

	nn, err := NewNeuralNetwork([]int{2, 2, 1})
	if err != nil {
		t.Error(err)
	}

	i := mat.NewVecDense(2, []float64{1, 1})
	err = nn.SetInput(i)
	if err != nil {
		t.Error(err)
	}

	d := mat.NewVecDense(1, []float64{desiredAnswer})

	for i := 0; i < 1000; i++ {
		nn.Forward()
		nn.Back(d)
		nn.Nudge(1)
	}

	nn.Forward()
	if nn.GetOutput().AtVec(0) != actualAnswer {
		t.Error(nn.GetOutput().AtVec(0))
	}
}
