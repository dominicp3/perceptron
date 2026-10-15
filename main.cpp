#include "network.hpp"
#include "vec.hpp"
#include "activation.hpp"
#include "data.hpp"
#include <iostream>
#include <random>
#include <utility>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <numeric>
#include <cmath>

std::random_device rd;

namespace {
	/* read a 4-byte big-endian unsigned integer */
	std::uint32_t readU32(std::ifstream &f)
	{
		unsigned char b[4];
		f.read(reinterpret_cast<char *>(b), 4);
		if (!f) {
			throw std::runtime_error("unexpected end of file while reading header");
		}
		return (std::uint32_t(b[0]) << 24) | (std::uint32_t(b[1]) << 16) |
		       (std::uint32_t(b[2]) << 8)  |  std::uint32_t(b[3]);
	}
}

std::vector<std::vector<double>> loadImages(const std::string &path)
{
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		throw std::runtime_error("cannot open " + path);
	}

	if (readU32(f) != 0x00000803) {
		throw std::runtime_error("bad magic number in image file (is it decompressed?)");
	}

	std::uint32_t count = readU32(f);
	std::uint32_t rows = readU32(f);
	std::uint32_t cols = readU32(f);
	std::size_t size = static_cast<std::size_t>(rows) * cols;

	std::vector<std::vector<double>> images(count, std::vector<double>(size));
	std::vector<unsigned char> buf(size);

	for (std::uint32_t n = 0; n < count; ++n) {
		f.read(reinterpret_cast<char *>(buf.data()), size);
		if (!f) {
			throw std::runtime_error("unexpected end of image data");
		}
		for (std::size_t i = 0; i < size; ++i) {
			images[n][i] = buf[i] / 255.0;
		}
	}

	return images;
}

std::vector<int> loadLabels(const std::string &path)
{
	std::ifstream f(path, std::ios::binary);
	if (!f) {
		throw std::runtime_error("cannot open " + path);
	}

	if (readU32(f) != 0x00000801) {
		throw std::runtime_error("bad magic number in label file (is it decompressed?)");
	}

	std::uint32_t count = readU32(f);

	std::vector<unsigned char> buf(count);
	f.read(reinterpret_cast<char *>(buf.data()), count);
	if (!f) {
		throw std::runtime_error("unexpected end of label data");
	}

	return std::vector<int>(buf.begin(), buf.end());
}

double relu(double x)
{
	if (x > 0.0) {
		return x;
	}

	return 0.0;
}

double dRelu(double x)
{
	if (x > 0.0) {
		return 1.0;
	}

	return 0.0;
}

Dataset gaussian_blob(int n)
{
	std::random_device rd;
	std::mt19937_64 rng(rd());
	std::normal_distribution<double> noise(0.0, 0.5);   // sigma = 0.5

	std::vector<std::vector<double>> xs;
	std::vector<int> ks;
	for (int i = 0; i < n; ++i) {
		xs.push_back({-1.0 + noise(rng), -1.0 + noise(rng)});  ks.push_back(0);
		xs.push_back({ 1.0 + noise(rng),  1.0 + noise(rng)});  ks.push_back(1);
		xs.push_back({ -1.0 + noise(rng),  1.0 + noise(rng)});  ks.push_back(2);
	}

	return {xs, ks, 3};
}

Dataset xor_set()
{
	std::vector<std::vector<double>> dataset {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};

	std::vector<int> labels {0, 1, 1, 0};

	return {dataset, labels, 2};
}

int floor_inner(const std::vector<double> &a, const std::vector<double> &b)
{
	double x = 0.0;

	for (std::size_t i = 0; i < a.size(); ++i) {
		x += a[i] * b[i];
	}

	return (int) std::floor(x);
}

Dataset generate_data(int data_size)
{
	Dataset d {
		std::vector<std::vector<double>>(data_size),
		std::vector<int>(data_size),
		10
	};

	std::vector<double> ones(10, 1.0);

	std::mt19937_64 rng(rd());

        for (int i = 0; i < data_size; i++) {
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		std::vector<double> v(10);

		for (double &k : v) {
			k = dist(rng);
		}

		d.X[i] = v;
		d.y[i] = floor_inner(v, ones);
        }

        return d;
}

void train(Perceptron &net, const Dataset &d, int batch_size, double lr, int epochs)
{
	const auto &[X, y, num_classes] = d;
	
	std::mt19937_64 rng(rd());

	int N = (int) d.X.size();

	std::vector<std::size_t> idx(N);
	std::iota(idx.begin(), idx.end(), 0);

	for (int epoch = 0; epoch < epochs; ++epoch) {
		std::shuffle(idx.begin(), idx.end(), rng);

		for (int start = 0; start < N; start += batch_size) {

			int end = std::min(start + batch_size, N);

			for (int i = start; i < end; ++i) {
				net.backpropagate(X[idx[i]], y[idx[i]]);
			}

			net.applyWeights(lr / (end - start));

		}
	}
}

void example_stats(Perceptron &net, const std::vector<double> &x, int k)
{
	std::cout << "probs: " << net.probabilities(x) << ", ";
	std::cout << "pred: " << net.predict(x) << ", ";
	std::cout << "true: " << k << "\n";
}

double test(Perceptron &net, const Dataset &d)
{
	const auto &[X, y, num_classes] = d;

	int dataset_size = (int) d.X.size();

	double hits = 0.0;

	for (int i = 0; i < dataset_size; ++i) {

		if (net.predict(X[i]) == y[i]) {
			++hits;
		} else {
			example_stats(net, X[i], y[i]);
		}
	}

	return hits / dataset_size;
}

void fuck()
{
	int n_train = 100000;
	Dataset d_train = generate_data(n_train);

	int n_test = 100;
	Dataset d_test = generate_data(n_test);

	Perceptron net(10, 10, {128, 128}, {relu, dRelu});

	train(net, d_train, 100, 0.5, 3);
	double hit_rate = test(net, d_test);

	std::cout << "hit rate: " << 100 * hit_rate << '%' << '\n';
}

Dataset mnist_train()
{
	return {
		loadImages("mnist/train-images.idx3-ubyte"),
		loadLabels("mnist/train-labels.idx1-ubyte"),
		10
	};
}

Dataset mnist_test()
{
	return {
		loadImages("mnist/t10k-images.idx3-ubyte"),
		loadLabels("mnist/t10k-labels.idx1-ubyte"),
		10
	};
}

void mnist_toy()
{
	auto [X_mnist_train, y_mnist_train, n_classes_mnist_train] = mnist_train();
	auto [X_mnist_test, y_mnist_test, n_classes_mnist_test] = mnist_test();

	Perceptron net((int) X_mnist_train.front().size(), n_classes_mnist_train, {256, 256}, {relu, dRelu});

	int epochs = 3;
	double learning_rate = 0.1;
	int batch_size = 128;

	net.fit(X_mnist_train, y_mnist_train, epochs, learning_rate, batch_size);

	double hit_rate = test(net, {X_mnist_test, y_mnist_test, n_classes_mnist_test});

	std::cout << "hit rate: " << 100 * hit_rate << '%' << '\n';
}

int main()
{
	mnist_toy();
}
