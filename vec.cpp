#include "vec.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

std::vector<double> &operator+=(std::vector<double> &a, const std::vector<double> &b)
{
	if (a.size() != b.size()) {
		throw std::invalid_argument("vector sizes differ");
	}

	for (std::size_t i = 0; i < a.size(); ++i) {
		a[i] += b[i];
	}

	return a;
}

std::vector<double> &addScaled(std::vector<double> &a, const std::vector<double> &b, double lambda)
{
	if (a.size() != b.size()) {
		throw std::invalid_argument("vector sizes differ");
	}

	for (std::size_t i = 0; i < a.size(); ++i) {
		a[i] += lambda * b[i];
	}

	return a;
}

std::vector<double> &hadamardInplace(std::vector<double> &a, const std::vector<double> &b)
{
	if (a.size() != b.size()) {
		throw std::invalid_argument("vector sizes differ");
	}

	for (std::size_t i = 0; i < a.size(); ++i) {
		a[i] *= b[i];
	}

	return a;
}

double crossEntropy(const std::vector<double> &z, int k)
{
	if (k < 0 || k >= (int) z.size()) {
		throw std::invalid_argument("crossEntropy: invalid k");
	}

	double m = *std::max_element(z.begin(), z.end());
	double s = 0.0;
	for (double v : z) {
		s += std::exp(v - m);
	}

	return -z[k] + m + std::log(s);
}

std::ostream &operator<<(std::ostream &os, const std::vector<double> &v)
{
	os << '[';

	for (std::size_t i = 0; i < v.size() - 1; ++i) {
		os << v[i] << ", ";
	}

	os << v.back() << "]";

	return os;
}
