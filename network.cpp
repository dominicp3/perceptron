#include "network.hpp"
#include "vec.hpp"
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <fstream>
#include <iostream>

Perceptron::Perceptron(int n_input, int m_output, const std::vector<int> &layer_config, Activation act):
	n_input{n_input},
	m_output{m_output},
	act_{act}
{
	/* first hidden layer */
	layers_.push_back({n_input, layer_config.front()});

	/* hidden layers */
	for (std::size_t i = 0; i < layer_config.size() - 1; ++i) {
		layers_.push_back({layer_config[i], layer_config[i + 1]});
	}

	/* last layer */
	layers_.push_back({layer_config.back(), m_output});
}

Perceptron::Perceptron(const std::string &file)
{
	std::ifstream in(file, std::ios::binary);

	if (!in.is_open()) {
		throw std::invalid_argument("loadFromFile: invalid file");
	}

	int rows;
	int cols;
	int sz;

	std::vector<Layer> layers;

	while (true) {
		in.read(reinterpret_cast<char*>(&rows), sizeof rows);

		if (in.gcount() == 0) {
			if (in.eof()) {
				break;
			} else {
				throw std::invalid_argument("error in reading file");
			}
		}

		in.read(reinterpret_cast<char*>(&cols), sizeof cols);

		sz = rows * cols;

		std::vector<float> W_arr(sz);
		in.read(reinterpret_cast<char*>(W_arr.data()), sz * sizeof (float));

		Matrix W(rows, cols, std::move(W_arr));

		std::vector<float> b(rows);
		in.read(reinterpret_cast<char*>(b.data()), rows * sizeof (float));

		Layer layer(cols, rows);
		layer.W = std::move(W);
		layer.b = std::move(b);

		layers.emplace_back(std::move(layer));
	}

	if (layers.empty()) {
		return;
	}

	n_input = layers.front().W.numCols();
	m_output = layers.back().W.numRows();
	act_ = {relu, dRelu};
	layers_ = std::move(layers);
}

void Perceptron::fit(const std::vector<std::vector<float>> &X, const std::vector<int> &y, int epochs, float learning_rate, int batch_size)
{
	if (X.size() != y.size()) {
		throw std::invalid_argument("fit: X and y have different sizes");
	}

	int N = (int) X.size();

	std::random_device rd;
	std::mt19937_64 rng(rd());

	std::vector<std::size_t> idx(N);
	std::iota(idx.begin(), idx.end(), 0);

	for (int epoch = 0; epoch < epochs; ++epoch) {
		std::shuffle(idx.begin(), idx.end(), rng);

		for (int start = 0; start < N; start += batch_size) {
			int end = std::min(start + batch_size, N);

			for (int i = start; i < end; ++i) {
				backpropagate(X[idx[i]], y[idx[i]]);
			}

			applyWeights(learning_rate / (end - start));
		}
	}
}

void Perceptron::forward(const std::vector<float> &x)
{
	layers_.front().input = x;

	for (auto it = layers_.begin(); it != layers_.end() - 1; ++it) {
		/* set affine output z*/
		it->W.affineTransform(it->input, it->b, it->z);

		/* set activation a */
		for (std::size_t i = 0; i < it->z.size(); ++i) {
			it->a[i] = act_.f(it->z[i]);
		}

		(it + 1)->input = it->a;
	}

	/* set affine output z for final layer*/
	auto last_it = layers_.end() - 1;
	last_it->W.affineTransform(last_it->input, last_it->b, last_it->z);
}

int Perceptron::predict(const std::vector<float> &x)
{
	forward(x);
	return std::max_element(layers_.back().z.begin(), layers_.back().z.end()) - layers_.back().z.begin();
}

const std::vector<float> Perceptron::probabilities(const std::vector<float> &x)
{
	forward(x);
	return softmax(layers_.back().z);
}

const std::vector<float> Perceptron::logits(const std::vector<float> &x)
{
	forward(x);
	return layers_.back().z;
}

float Perceptron::loss(const std::vector<float> &x, int k)
{
	if (k < 0 || k >= m_output) {
		throw std::invalid_argument("loss: k invalid index");
	}

	forward(x);
	return crossEntropy(layers_.back().z, k);
}

float Perceptron::backpropagate(const std::vector<float> &x, int k)
{
	if ((int) x.size() != n_input || k < 0 || k >= m_output) {
		throw std::invalid_argument("backpropagate: k invalid index");
	}

	forward(x);
	backward(k);

	return crossEntropy(layers_.back().z, k);
}

void Perceptron::applyWeights(float alpha)
{
	for (Layer &layer : layers_) {
		/* perform W += alpha * Wg */
		layer.W.addScaled(-alpha, layer.Wg);

		/* perform b += alpha * bg */
		addScaled(layer.b, layer.bg, -alpha);
	}

	clearGradients();
}

void Perceptron::clearGradients()
{
	for (Layer &layer : layers_) {
		layer.Wg.zero();
		std::fill(layer.bg.begin(), layer.bg.end(), 0.0);
	}
}

void Perceptron::saveToFile(const std::string &filename) const
{
	std::ofstream out(filename, std::ios::binary);

	if (!out.is_open()) {
		throw std::invalid_argument("saveToFile: invalid file");
	}

	for (const Layer &layer : layers_) {
		auto [rows, cols] = layer.W.shape();
		const auto *W_data = layer.W.data();
		const auto *b_data = layer.b.data();

		out.write(reinterpret_cast<const char*>(&rows), sizeof rows);
		out.write(reinterpret_cast<const char*>(&cols), sizeof cols);

		int sz = rows * cols;

		out.write(reinterpret_cast<const char*>(W_data), sz * sizeof *W_data);
		out.write(reinterpret_cast<const char*>(b_data), rows * sizeof *b_data);
	}
}

bool Perceptron::equalParams(const Perceptron &other) const
{
	if (layers_.size() != other.layers_.size()) {
		std::cout << "layer size\n";
		return false;
	}

	for (std::size_t i = 0; i < layers_.size(); ++i) {
		if (layers_[i].W != other.layers_[i].W) {
			std::cout << "W: " << i << '\n';
			return false;
		}

		if (layers_[i].b != other.layers_[i].b) {
			std::cout << "b: " << i << '\n';
			return false;
		}
	}

	return true;
}

const Layer &Perceptron::getLayer(int layer) const
{
	return layers_.at(layer);
}

const Matrix &Perceptron::getWeights(int layer) const
{
	return layers_.at(layer).W;
}
const std::vector<float> &Perceptron::getBiases(int layer) const
{
	return layers_.at(layer).b;
}

int Perceptron::numLayers() const
{
	return (int) layers_.size();
}

std::vector<float> Perceptron::softmax(const std::vector<float>& x)
{
	std::vector<float> p(x.size());

	float m = *std::max_element(x.begin(), x.end());

	float d = 0.0;
	for (std::size_t i = 0; i < x.size(); ++i) {
		p[i] = std::exp(x[i] - m);
		d += p[i];
	}

	for (float &v : p) {
		v /= d;
	}

	return p;
}

void Perceptron::backward(int k)
{
	if (k < 0 || k >= m_output) {
		throw std::invalid_argument("backward: k invalid index");
	}

	auto last_it = layers_.end() - 1;

	last_it->delta = softmax(last_it->z);
	last_it->delta[k] -= 1.0;

	last_it->Wg.addOuter(last_it->delta, last_it->input);
	last_it->bg += last_it->delta;

	for (auto it = layers_.rbegin() + 1; it != layers_.rend(); ++it) {
		auto prev_it = it - 1;

		prev_it->W.transposeProduct(prev_it->delta, it->delta);

		for (std::size_t i = 0; i < it->delta.size(); ++i) {
			it->delta[i] *= act_.df(it->z[i]);
		}

		it->Wg.addOuter(it->delta, it->input);
		it->bg += it->delta;
	}
}
