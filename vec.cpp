#include "vec.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

std::vector<float> &operator+=(std::vector<float> &a, const std::vector<float> &b)
{
	if (a.size() != b.size()) {
		throw std::invalid_argument("vector sizes differ");
	}

	for (std::size_t i = 0; i < a.size(); ++i) {
		a[i] += b[i];
	}

	return a;
}

std::vector<float> &addScaled(std::vector<float> &a, const std::vector<float> &b, float lambda)
{
	if (a.size() != b.size()) {
		throw std::invalid_argument("vector sizes differ");
	}

	for (std::size_t i = 0; i < a.size(); ++i) {
		a[i] += lambda * b[i];
	}

	return a;
}

std::vector<float> &hadamardInplace(std::vector<float> &a, const std::vector<float> &b)
{
	if (a.size() != b.size()) {
		throw std::invalid_argument("vector sizes differ");
	}

	for (std::size_t i = 0; i < a.size(); ++i) {
		a[i] *= b[i];
	}

	return a;
}

float crossEntropy(const std::vector<float> &z, int k)
{
	if (k < 0 || k >= (int) z.size()) {
		throw std::invalid_argument("crossEntropy: invalid k");
	}

	float m = *std::max_element(z.begin(), z.end());
	float s = 0.0;
	for (float v : z) {
		s += std::exp(v - m);
	}

	return -z[k] + m + std::log(s);
}

std::ostream &operator<<(std::ostream &os, const std::vector<float> &v)
{
	os << '[';

	for (std::size_t i = 0; i < v.size() - 1; ++i) {
		os << v[i] << ", ";
	}

	os << v.back() << "]";

	return os;
}
