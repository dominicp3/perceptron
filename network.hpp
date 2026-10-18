#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <string>
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
	std::vector<float> input; // input from previous layer's activation

	Matrix W; // weight matrix
	std::vector<float> b; // bias vector

	std::vector<float> z; // affine output
	std::vector<float> a; // activation vector

	/*
		backpropagation space
	*/
	std::vector<float> delta; // error vector

	Matrix Wg; // accumulator matrix to hold weight gradient
	std::vector<float> bg; // accumulator vector to hold bias gradient
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
	Perceptron(int n_input, int m_output, const std::vector<int> &layer_config, Activation act = {relu, dRelu});

	/*
		load a perceptron from a config file
	*/
	Perceptron(const std::string &file);

	void fit(const std::vector<std::vector<float>> &X, const std::vector<int> &y, int epochs = 2, float learning_rate=0.01, int batch_size = 1);

	/* forward pass through the network */
	void forward(const std::vector<float> &x);

	int predict(const std::vector<float> &x);

	const std::vector<float> probabilities(const std::vector<float> &x);

	const std::vector<float> logits(const std::vector<float> &x);

	/* loss per example */
	float loss(const std::vector<float> &x, int k);

	/* update the accumulators */
	float backpropagate(const std::vector<float> &x, int k);

	/* use the accumulated gradients to update weights, gradient descent */
	void applyWeights(float alpha);

	/* zero out the accumulators to ready for next update */
	void clearGradients();

	void saveToFile(const std::string &filename) const;

	bool equalParams(const Perceptron &other) const;
	const Layer &getLayer(int layer) const;
	const Matrix &getWeights(int layer) const;
	const std::vector<float> &getBiases(int layer) const;
	int numLayers() const;

private:
	std::vector<float> softmax(const std::vector<float> &x);

	void backward(int k);

	int n_input;
	int m_output;

	Activation act_;

	std::vector<Layer> layers_;
};
