package gonn

import (
	"math"
	"math/rand"
)

// Init is a function which receives a size and returns an initialized array
type Init func(size int) []float64

// Fx - F(x)
type Fx func(x float64) float64

// InitNormal initializes weights with normally-distributed values
func InitNormal(size int) []float64 {
	data := make([]float64, size)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	return data
}

// InitZero initalizes weights with zeros
func InitZero(size int) []float64 {
	return nil
}

// Sigmoid is a logistic function
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// DSigmoid is a derivative of the Sigmoid
func DSigmoid(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}
