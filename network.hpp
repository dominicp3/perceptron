#pragma once
#include <vector>
#include <cmath>
#include <random>
#include "matrix.hpp"
#include "activation.hpp"

struct Layer
{
	Layer(int in, int out):
		input(in),
		W(Matrix::random(out, in, -std::sqrt(6.0 / in), std::sqrt(6.0 / in))),
		b(out),
		z(out),
		a(out),
		delta(out),
		Wg(out, in),
		bg(out) {}

	/*
		forward pass space
	*/
	std::vector<double> input; // input from previous layer's activation

	Matrix W; // weight matrix
	std::vector<double> b; // bias vector

	std::vector<double> z; // affine output
	std::vector<double> a; // activation vector

	/*
		backpropagation space
	*/
	std::vector<double> delta; // error vector

	Matrix Wg; // accumulator matrix to hold weight gradient
	std::vector<double> bg; // accumulator vector to hold bias gradient
};

/*
	for now, assume the final output is softmax combined with cross entropy loss
*/
class Perceptron
{
public:
	/*
		n_input: size of input vector
		m_output: size of output vector
		layer_config: vector of int specifying the input size of each hidden layer
	*/
	Perceptron(int n_input, int m_output, const std::vector<int> &layer_config, Activation act);

	void fit(const std::vector<std::vector<double>> &X, const std::vector<int> &y, int epochs = 2, double learning_rate=0.01, int batch_size = 1);

	/* forward pass through the network */
	void forward(const std::vector<double> &x);

	int predict(const std::vector<double> &x);

	const std::vector<double> probabilities(const std::vector<double> &x);

	const std::vector<double> logits(const std::vector<double> &x);

	/* loss per example */
	double loss(const std::vector<double> &x, int k);

	/* update the accumulators */
	double backpropagate(const std::vector<double> &x, int k);

	/* use the accumulated gradients to update weights, gradient descent */
	void applyWeights(double alpha);

	/* zero out the accumulators to ready for next update */
	void clearGradients();

	const Layer &getLayer(int layer) const;
	const Matrix &getWeights(int layer) const;
	const std::vector<double> &getBiases(int layer) const;
	int numLayers() const;

private:
	std::vector<double> softmax(const std::vector<double> &x);

	void backward(int k);

	int n_input;
	int m_output;

	Activation act_;

	std::vector<Layer> layers_;

	std::random_device rd_;
};
