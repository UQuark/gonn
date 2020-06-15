package gonn

import (
	"gonum.org/v1/gonum/mat"
)

// Back performs a back propagation
func (nn *NeuralNetwork) Back(desired *mat.VecDense) {
	var prevDelta *mat.VecDense

	for i := nn.transitionCount - 1; i >= 0; i-- {
		layerSize, _ := nn.raw[i].Dims()

		delta := mat.NewVecDense(layerSize, nil)

		if i == nn.transitionCount-1 {
			for j := 0; j < layerSize; j++ {
				delta.SetVec(j, (nn.value[i+1].AtVec(j)-desired.AtVec(j))*nn.derivative(nn.raw[i].AtVec(j)))
			}
		} else {
			prevLayerSize, _ := nn.raw[i+1].Dims()

			for j := 0; j < layerSize; j++ {
				var sum float64

				for k := 0; k < prevLayerSize; k++ {
					sum += nn.weight[i+1].At(k, j) * prevDelta.AtVec(k)
				}

				delta.SetVec(j, sum*nn.derivative(nn.raw[i].AtVec(j)))
			}
		}

		nn.deltaWeight[i].Apply(func(to, from int, v float64) float64 {
			return v + delta.AtVec(to)*nn.value[i].AtVec(from)
		}, nn.deltaWeight[i])

		nn.deltaBias[i].AddVec(nn.deltaBias[i], delta)

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
		nn.deltaBias[i].ScaleVec(scalar, nn.deltaBias[i])

		nn.weight[i].Sub(nn.weight[i], nn.deltaWeight[i])
		nn.bias[i].SubVec(nn.bias[i], nn.deltaBias[i])

		nn.deltaWeight[i].Zero()
		nn.deltaBias[i].Zero()
	}
}
