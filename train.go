package gonn

import (
	"gonum.org/v1/gonum/mat"
)

// Back performs a back propagation
func (nn *NeuralNetwork) Back(desired *mat.VecDense) {
	var prevDelta []float64

	for i := nn.transitionCount - 1; i >= 0; i-- {
		layerSize, _ := nn.raw[i].Dims()

		delta := make([]float64, layerSize)

		if i == nn.transitionCount-1 {
			for j := 0; j < layerSize; j++ {
				delta[j] = (nn.value[i+1].AtVec(j) - desired.AtVec(j)) * nn.derivative(nn.raw[i].AtVec(j))
			}
		} else {
			prevLayerSize, _ := nn.raw[i+1].Dims()

			for j := 0; j < layerSize; j++ {
				var sum float64

				for k := 0; k < prevLayerSize; k++ {
					sum += nn.weight[i].At(k, j) * prevDelta[k]
				}

				delta[j] = sum * nn.derivative(nn.raw[i].AtVec(j))
			}
		}

		nn.deltaWeight[i].Apply(func(to, from int, v float64) float64 {
			return v + delta[to]*nn.value[i].AtVec(from)
		}, nn.deltaWeight[i])
		prevDelta = delta
	}

	nn.backPasses++
}

// Nudge changes network's weights based on back propagation
func (nn *NeuralNetwork) Nudge(learningRate float64) {
	scalar := learningRate / float64(nn.backPasses)
	nn.backPasses = 0

	for i := 0; i < nn.transitionCount; i++ {
		nn.deltaWeight[i].Scale(scalar, nn.deltaWeight[i])
		nn.weight[i].Sub(nn.weight[i], nn.deltaWeight[i])
		nn.deltaWeight[i].Zero()
	}
}
