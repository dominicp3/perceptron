#include "network.hpp"
#include "vec.hpp"
#include <algorithm>
#include <stdexcept>
#include <set>

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

void Perceptron::fit(const std::vector<std::vector<double>> &X, const std::vector<int> &y, int epochs, double learning_rate, int batch_size)
{
	if (X.size() != y.size()) {
		throw std::invalid_argument("fit: X and y have different sizes");
	}
	
	std::mt19937_64 rng(rd_());

	int N = (int) X.size();

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

void Perceptron::forward(const std::vector<double> &x)
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

int Perceptron::predict(const std::vector<double> &x)
{
	forward(x);
	return std::max_element(layers_.back().z.begin(), layers_.back().z.end()) - layers_.back().z.begin();
}

const std::vector<double> Perceptron::probabilities(const std::vector<double> &x)
{
	forward(x);
	return softmax(layers_.back().z);
}

const std::vector<double> Perceptron::logits(const std::vector<double> &x)
{
	forward(x);
	return layers_.back().z;
}

double Perceptron::loss(const std::vector<double> &x, int k)
{
	if (k < 0 || k >= m_output) {
		throw std::invalid_argument("loss: k invalid index");
	}

	forward(x);
	return crossEntropy(layers_.back().z, k);
}

double Perceptron::backpropagate(const std::vector<double> &x, int k)
{
	if ((int) x.size() != n_input || k < 0 || k >= m_output) {
		throw std::invalid_argument("backpropagate: k invalid index");
	}

	forward(x);
	backward(k);

	return crossEntropy(layers_.back().z, k);
}

void Perceptron::applyWeights(double alpha)
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

const Layer &Perceptron::getLayer(int layer) const
{
	return layers_.at(layer);
}

const Matrix &Perceptron::getWeights(int layer) const
{
	return layers_.at(layer).W;
}
const std::vector<double> &Perceptron::getBiases(int layer) const
{
	return layers_.at(layer).b;
}

int Perceptron::numLayers() const
{
	return (int) layers_.size();
}

std::vector<double> Perceptron::softmax(const std::vector<double>& x)
{
	std::vector<double> p(x.size());

	double m = *std::max_element(x.begin(), x.end());

	double d = 0.0;
	for (std::size_t i = 0; i < x.size(); ++i) {
		p[i] = std::exp(x[i] - m);
		d += p[i];
	}

	for (double &v : p) {
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
