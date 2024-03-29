package gonn

import (
	"errors"

	"gonum.org/v1/gonum/mat"
)

var (
	// ErrSizeMismatch - data size mismatch
	ErrSizeMismatch = errors.New("data size mismatch")
)

// SetInput sets input values. Values are copied.
func (nn *NeuralNetwork) SetInput(data *mat.VecDense) error {
	input := nn.value[0]

	if input.Len() != data.Len() {
		return ErrSizeMismatch
	}

	input.CopyVec(data)

	return nil
}

// GetOutput returns output values. Values are copied.
func (nn *NeuralNetwork) GetOutput() *mat.VecDense {
	output := nn.value[len(nn.value)-1]
	data := mat.NewVecDense(output.Len(), nil)
	data.CloneVec(output)
	return data
}

// Forward performs a forward propagation
func (nn *NeuralNetwork) Forward() {
	for i := 0; i < nn.transitionCount; i++ {
		nn.raw[i].MulVec(nn.weight[i], nn.value[i])
		nn.raw[i].AddVec(nn.raw[i], nn.bias[i])
		for j := 0; j < nn.value[i+1].Len(); j++ {
			nn.value[i+1].SetVec(j, nn.activation(nn.raw[i].AtVec(j)))
		}
	}
}
