package gonn

import (
	"testing"
)

func TestNNCreation(t *testing.T) {
	_, err := NewNeuralNetwork([]int{2, 1, 2})
	if err != nil {
		t.Error(err)
	}

	_, err = NewNeuralNetwork([]int{2, 0, 2})
	if err != ErrTooSmallLayer {
		t.Error(err)
	}

	_, err = NewNeuralNetwork([]int{2})
	if err != ErrTooFewLayers {
		t.Error(err)
	}
}
