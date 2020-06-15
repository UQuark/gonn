package gonn

import (
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestForward(t *testing.T) {
	rand.Seed(0)

	const (
		actualAnswer = 0.536850925930091
	)

	nn, err := NewNeuralNetwork([]int{2, 2, 1}, InitNormal, Sigmoid, DSigmoid)
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
