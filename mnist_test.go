package gonn

import (
	"fmt"
	"io"
	"math/rand"
	"os"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func mnist(labelsFilename, imagesFilename string, count int) (labels []byte, images [][]float64, err error) {
	f, err := os.Open(labelsFilename)
	if err != nil {
		return
	}
	defer f.Close()
	labels = make([]byte, count)
	f.Seek(8, io.SeekStart)
	for i := 0; i < count; i++ {
		b := make([]byte, 1)
		f.Read(b)
		labels[i] = b[0]
	}
	f.Close()

	f, err = os.Open(imagesFilename)
	if err != nil {
		return
	}
	defer f.Close()
	images = make([][]float64, count)
	f.Seek(16, io.SeekStart)
	for i := 0; i < count; i++ {
		b := make([]byte, 28*28)
		f.Read(b)
		images[i] = make([]float64, 28*28)
		for j := 0; j < 28*28; j++ {
			images[i][j] = float64(b[j]) / 255.0
		}
	}

	return
}

func makeDesiredVector(label byte) *mat.VecDense {
	data := make([]float64, 10)
	data[label] = 1
	return mat.NewVecDense(10, data)
}

func squaredErr(a, b *mat.VecDense) float64 {
	var err float64
	for i := 0; i < a.Len(); i++ {
		sum := a.AtVec(i) + b.AtVec(i)
		err += sum * sum
	}
	return err
}

func check(answer *mat.VecDense, correct byte) (valid bool, confidence float64) {
	maxPos := 0
	max := answer.AtVec(0)

	for i := 0; i < answer.Len(); i++ {
		if max < answer.AtVec(i) {
			max = answer.AtVec(i)
			maxPos = i
		}
	}

	confidence = max
	valid = (maxPos == int(correct))

	return
}

func TestMNIST(t *testing.T) {
	rand.Seed(0)

	nn, err := NewNeuralNetwork([]int{28 * 28, 16, 10}, InitNormal, Sigmoid, DSigmoid)
	if err != nil {
		t.Error(err)
	}

	const (
		MNISTTrainingLabels = "mnist/train-labels-idx1-ubyte"
		MNISTTrainingImages = "mnist/train-images-idx3-ubyte"
		MNISTTrainingLength = 60000

		MaxLoop        = 155000
		DesiredPercent = 85
	)

	labels, images, err := mnist(MNISTTrainingLabels, MNISTTrainingImages, MNISTTrainingLength)
	if err != nil {
		t.Error(err)
	}

	correct := 0
	total := 0
	i := 0
	lastPercent := 0

	for lastPercent < DesiredPercent && total <= MaxLoop {
		i %= MNISTTrainingLength

		input := mat.NewVecDense(28*28, images[i])
		nn.SetInput(input)
		nn.Forward()

		output := nn.GetOutput()
		valid, _ := check(output, labels[i])
		if valid {
			correct++
		}
		total++

		percent := 100 * correct / total

		if percent != lastPercent {
			lastPercent = percent
			fmt.Printf("Run: %d\tCorrect: %d%%\n", total, percent)
		}

		desired := makeDesiredVector(labels[i])
		nn.Back(desired)
		nn.Nudge(.5)

		i++
	}

	if lastPercent < DesiredPercent {
		t.Error()
	}
}
