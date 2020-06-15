package gonn

import (
	"errors"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

var (
	// ErrTooFewLayers - at least 2 layers must be present in a NeuralNetwork
	ErrTooFewLayers = errors.New("At least 2 layers must be present in a NeuralNetwork")
	// ErrTooSmallLayer - at least 1 neuron must be present on a layer
	ErrTooSmallLayer = errors.New("At least 1 neuron must be present on a layer")
)

// NeuralNetwork represents a multi-layer perceptron
type NeuralNetwork struct {
	raw, value          []*mat.VecDense
	weight, deltaWeight []*mat.Dense

	layerCount, transitionCount int
	backPasses                  int
}

func initNormal(size int) []float64 {
	data := make([]float64, size)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	return data
}

func initZero(size int) []float64 {
	return nil
}

// NewNeuralNetwork creates a new NeuralNetwork given layers' sizes
func NewNeuralNetwork(layers []int) (nn *NeuralNetwork, err error) {
	if len(layers) < 2 {
		err = ErrTooFewLayers
		return
	}

	nn = &NeuralNetwork{}

	nn.layerCount = len(layers)
	nn.transitionCount = nn.layerCount - 1

	nn.weight = make([]*mat.Dense, nn.transitionCount)
	nn.deltaWeight = make([]*mat.Dense, nn.transitionCount)
	nn.raw = make([]*mat.VecDense, nn.transitionCount)
	nn.value = make([]*mat.VecDense, nn.layerCount)

	for i := 0; i < nn.transitionCount; i++ {
		if layers[i] < 1 || layers[i+1] < 1 {
			nn = nil
			err = ErrTooSmallLayer
			return
		}

		nn.weight[i] = mat.NewDense(layers[i+1], layers[i], initNormal(layers[i+1]*layers[i]))
		nn.deltaWeight[i] = mat.NewDense(layers[i+1], layers[i], initZero(layers[i+1]*layers[i]))
		nn.raw[i] = mat.NewVecDense(layers[i+1], nil)
	}

	for i := 0; i < nn.layerCount; i++ {
		nn.value[i] = mat.NewVecDense(layers[i], nil)
	}

	return
}
