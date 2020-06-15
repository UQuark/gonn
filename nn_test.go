package gonn

import (
	"testing"
)

func TestNNCreation(t *testing.T) {
	_, err := NewNeuralNetwork([]int{2, 1, 2}, InitNormal, Sigmoid, DSigmoid)
	if err != nil {
		t.Error(err)
	}

	_, err = NewNeuralNetwork([]int{2, 0, 2}, InitNormal, Sigmoid, DSigmoid)
	if err != ErrTooSmallLayer {
		t.Error(err)
	}

	_, err = NewNeuralNetwork([]int{2}, InitNormal, Sigmoid, DSigmoid)
	if err != ErrTooFewLayers {
		t.Error(err)
	}
}
