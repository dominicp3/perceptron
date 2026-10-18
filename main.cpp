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
#include <fstream>
#include <string>

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

std::vector<std::vector<float>> loadImages(const std::string &path)
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

	std::vector<std::vector<float>> images(count, std::vector<float>(size));
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

Dataset xor_set()
{
	std::vector<std::vector<float>> dataset {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};

	std::vector<int> labels {0, 1, 1, 0};

	return {dataset, labels, 2};
}

int floor_inner(const std::vector<float> &a, const std::vector<float> &b)
{
	float x = 0.0;

	for (std::size_t i = 0; i < a.size(); ++i) {
		x += a[i] * b[i];
	}

	return (int) std::floor(x);
}

Dataset generate_data(int data_size)
{
	Dataset d {
		std::vector<std::vector<float>>(data_size),
		std::vector<int>(data_size),
		10
	};

	std::vector<float> ones(10, 1.0);

	std::mt19937_64 rng(rd());

        for (int i = 0; i < data_size; i++) {
		std::uniform_real_distribution<float> dist(0.0, 1.0);

		std::vector<float> v(10);

		for (float &k : v) {
			k = dist(rng);
		}

		d.X[i] = v;
		d.y[i] = floor_inner(v, ones);
        }

        return d;
}

void train(Perceptron &net, const Dataset &d, int batch_size, float lr, int epochs)
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

void example_stats(Perceptron &net, const std::vector<float> &x, int k)
{
	std::cout << "probs: " << net.probabilities(x) << ", ";
	std::cout << "pred: " << net.predict(x) << ", ";
	std::cout << "true: " << k << "\n";
}

float test(Perceptron &net, const Dataset &d)
{
	const auto &[X, y, num_classes] = d;

	int dataset_size = (int) d.X.size();

	float hits = 0.0;

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
	float hit_rate = test(net, d_test);

	std::cout << "hit rate: " << 100 * hit_rate << '%' << '\n';
}

Dataset mnist_trainset()
{
	return {
		loadImages("mnist/train-images.idx3-ubyte"),
		loadLabels("mnist/train-labels.idx1-ubyte"),
		10
	};
}

Dataset mnist_testset()
{
	return {
		loadImages("mnist/t10k-images.idx3-ubyte"),
		loadLabels("mnist/t10k-labels.idx1-ubyte"),
		10
	};
}

Perceptron mnist_train(int epochs, float lr, int b, const std::vector<int> &config)
{
	auto [X_mnist_train, y_mnist_train, n_classes_mnist_train] = mnist_trainset();

	Perceptron net((int) X_mnist_train.front().size(), n_classes_mnist_train, config, {relu, dRelu});

	net.fit(X_mnist_train, y_mnist_train, epochs, lr, b);

	return net;
}

void mnist_test(Perceptron &net)
{
	auto [X_mnist_test, y_mnist_test, n_classes_mnist_test] = mnist_testset();
	float hit_rate = test(net, {X_mnist_test, y_mnist_test, n_classes_mnist_test});
	std::cout << "hit rate: " << 100 * hit_rate << '%' << '\n';
}

std::vector<float> mnist_single(std::string filename)
{
	std::ifstream file(filename);

	if (!file) {
		std::cerr << "Failed to open pixels.txt\n";
	}

	std::vector<float> pixels;
	float value;

	while (file >> value) {
		pixels.push_back(value);
	}

	if (pixels.size() != 784) {
		std::cerr << "Expected 784 pixels, got " << pixels.size() << '\n';
	}

	std::cout << "Loaded " << pixels.size() << " pixels\n";

	return pixels;
}

int main()
{
	// int epochs = 20;
	// float learning_rate = 0.005;
	// int batch_size = 64;
	// std::vector<int> config {256, 256};
	// std::string mnist_file = "mnist_weights/test";
	// Perceptron net = mnist_train(epochs, learning_rate, batch_size, config);
	// net.saveToFile(mnist_file);


	// Perceptron net("mnist_weights/test");
	// mnist_test(net);


	// Perceptron net_256("mnist_epoch20_lr005_b64_config256.256");
	// Perceptron net_512("mnist_epoch40_lr005_b64_config512.512");

	// std::cout << "512,512:\n";

	// for (int k = 0; k < 10; ++k) {
	// 	std::string digit_file = "digits/";
	// 	digit_file += std::to_string(k);
	// 	digit_file += ".png.txt";

	// 	std::ifstream file(digit_file);
	// 	std::vector<float> values;
	// 	float x;

	// 	while (file >> x) {
	// 		values.push_back(x);
	// 	}

	// 	example_stats(net_512, values, k);
	// }

	// std::cout << "\n256,256:\n";

	// for (int k = 0; k < 10; ++k) {
	// 	std::string digit_file = "digits/";
	// 	digit_file += std::to_string(k);
	// 	digit_file += ".png.txt";

	// 	std::ifstream file(digit_file);
	// 	std::vector<float> values;
	// 	float x;

	// 	while (file >> x) {
	// 		values.push_back(x);
	// 	}

	// 	example_stats(net_256, values, k);
	// }


	// std::cout << "\n512, 128:\n";

	// for (int k = 0; k < 10; ++k) {
	// 	std::string digit_file = "digits/";
	// 	digit_file += std::to_string(k);
	// 	digit_file += ".png.txt";

	// 	std::ifstream file(digit_file);
	// 	std::vector<float> values;
	// 	float x;

	// 	while (file >> x) {
	// 		values.push_back(x);
	// 	}

	// 	example_stats(net, values, k);
	// }
}


