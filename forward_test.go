package gonn

import (
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestForward(t *testing.T) {
	const (
		actualAnswer = 0.7793750625809166
		seed         = 0
	)

	rand.Seed(seed)

	nn, err := NewNeuralNetwork([]int{2, 2, 1})
	if err != nil {
		t.Error(err)
	}

	i := mat.NewVecDense(1, []float64{1})
	err = nn.SetInput(i)
	if err != ErrSizeMismatch {
		t.Error(err)
	}

	i = mat.NewVecDense(2, []float64{1, 1})
	err = nn.SetInput(i)
	if err != nil {
		t.Error(err)
	}
	nn.Forward()
	o := nn.GetOutput()
	if o.AtVec(0) != actualAnswer {
		t.Error(o.AtVec(0))
	}
}
