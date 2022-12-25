package gonn

import (
	"errors"

	"gonum.org/v1/gonum/mat"
)

var (
	// ErrTooFewLayers - at least 2 layers must be present in a NeuralNetwork
	ErrTooFewLayers = errors.New("at least 2 layers must be present in a NeuralNetwork")
	// ErrTooSmallLayer - at least 1 neuron must be present on a layer
	ErrTooSmallLayer = errors.New("at least 1 neuron must be present on a layer")
)

// NeuralNetwork represents a multi-layer perceptron
type NeuralNetwork struct {
	raw, value, bias, deltaBias []*mat.VecDense
	weight, deltaWeight         []*mat.Dense

	layerCount, transitionCount int
	backPasses                  int

	activation, derivative Fx
}

// NewNeuralNetwork creates a new NeuralNetwork given layers' sizes
func NewNeuralNetwork(layers []int, init Init, activation, derivative Fx) (nn *NeuralNetwork, err error) {
	if len(layers) < 2 {
		err = ErrTooFewLayers
		return
	}

	nn = &NeuralNetwork{}

	nn.activation = activation
	nn.derivative = derivative

	nn.layerCount = len(layers)
	nn.transitionCount = nn.layerCount - 1

	nn.weight = make([]*mat.Dense, nn.transitionCount)
	nn.deltaWeight = make([]*mat.Dense, nn.transitionCount)

	nn.raw = make([]*mat.VecDense, nn.transitionCount)

	nn.bias = make([]*mat.VecDense, nn.transitionCount)
	nn.deltaBias = make([]*mat.VecDense, nn.transitionCount)

	nn.value = make([]*mat.VecDense, nn.layerCount)

	for i := 0; i < nn.transitionCount; i++ {
		if layers[i] < 1 || layers[i+1] < 1 {
			nn = nil
			err = ErrTooSmallLayer
			return
		}

		nn.weight[i] = mat.NewDense(layers[i+1], layers[i], init(layers[i+1]*layers[i]))
		nn.deltaWeight[i] = mat.NewDense(layers[i+1], layers[i], nil)

		nn.raw[i] = mat.NewVecDense(layers[i+1], nil)

		nn.bias[i] = mat.NewVecDense(layers[i+1], init(layers[i+1]))
		nn.deltaBias[i] = mat.NewVecDense(layers[i+1], nil)
	}

	for i := 0; i < nn.layerCount; i++ {
		nn.value[i] = mat.NewVecDense(layers[i], nil)
	}

	return
}
