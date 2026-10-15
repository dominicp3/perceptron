#include "matrix.hpp"
#include <random>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace {
	std::random_device rd;
	std::mt19937_64 gen(rd());
}

Matrix::Matrix(int n_rows, int m_cols, double value):
	rows_{n_rows},
	cols_{m_cols},
	arr_(n_rows * m_cols, value) {}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> rows)
{
	if (rows.size() == 0) {
		throw std::invalid_argument("Matrix cannot be empty");
	}

	std::size_t n_rows = rows.size();
	std::size_t n_cols = rows.begin()->size();

	rows_ = n_rows;
	cols_ = n_cols;
	arr_ = std::vector<double>(n_rows * n_cols);

	std::size_t i = 0;
	for (const auto &row : rows) {

		if (row.size() != n_cols) {
			throw std::invalid_argument("Rows must have equal length");
		}

		std::size_t j = 0;
		for (double entry : row) {
			arr_[i * n_cols + j] = entry;
			++j;
		}

		++i;
	}
}

Matrix Matrix::identity(int n)
{
	Matrix m(n, n);

	for (int i = 0; i < n; ++i) {
		m.arr_[i * n + i] = 1;
	}

	return m;
}

Matrix Matrix::random(int n_rows, int m_cols, double lo, double hi)
{
	Matrix m(n_rows, m_cols);

	std::uniform_real_distribution<double> dist(lo, hi);

	for (double &x : m.arr_) {
		x = dist(gen);
	}

	return m;
}

std::pair<int, int> Matrix::shape() const
{
	return {rows_, cols_};
}

double Matrix::operator()(int i, int j) const
{
	return arr_[i * cols_ + j];
}

double& Matrix::operator()(int i, int j)
{
	return arr_[i * cols_ + j];
}

Matrix Matrix::operator+(const Matrix &other) const
{
	if (rows_ != other.rows_ || cols_ != other.cols_) {
		throw std::invalid_argument("dimension mismatch");
        }
	
	Matrix m(rows_, cols_);

	for (std::size_t i = 0; i < arr_.size(); ++i) {
		m.arr_[i] = arr_[i] + other.arr_[i];
	}

	return m;
}

Matrix &Matrix::operator+=(const Matrix &other)
{
	if (rows_ != other.rows_ || cols_ != other.cols_) {
		throw std::invalid_argument("dimension mismatch");
        }

	for (std::size_t i = 0; i < arr_.size(); ++i) {
		arr_[i] += other.arr_[i];
	}

	return *this;
}

Matrix Matrix::operator-(const Matrix &other) const
{
	if (rows_ != other.rows_ || cols_ != other.cols_) {
		throw std::invalid_argument("dimension mismatch");
        }

	Matrix m(rows_, cols_);

	for (std::size_t i = 0; i < arr_.size(); ++i) {
		m.arr_[i] = arr_[i] - other.arr_[i];
	}

	return m;
}

Matrix &Matrix::operator-=(const Matrix &other)
{
	if (rows_ != other.rows_ || cols_ != other.cols_) {
		throw std::invalid_argument("dimension mismatch");
        }

	for (std::size_t i = 0; i < arr_.size(); ++i) {
		arr_[i] -= other.arr_[i];
	}

	return *this;
}

Matrix Matrix::operator*(double lambda) const
{
	Matrix m(rows_, cols_);

	for (std::size_t i = 0; i < arr_.size(); ++i) {
		m.arr_[i] = lambda * arr_[i];
	}

	return m;
}

Matrix Matrix::operator*(const Matrix &other) const
{
	if (cols_ != other.rows_) {
		throw std::invalid_argument("dimension mismatch");
	}

	Matrix m(rows_, other.cols_);

	for (int i = 0; i < rows_; ++i) {
		for (int j = 0; j < other.cols_; ++j) {
			double entry = 0.0;
			for (int k = 0; k < cols_; ++k) {
				entry += arr_[i * cols_ + k] * other.arr_[k * other.cols_ + j];
			}
			m.arr_[i * m.cols_ + j] = entry;
		}
	}

	return m;
}

std::vector<double> Matrix::operator*(const std::vector<double> &x) const
{
	if (cols_ != (int) x.size()) {
		throw std::invalid_argument("dimension mismatch");
	}

	std::vector<double> y(rows_);

	for (int i = 0; i < rows_; ++i) {
		double e = 0.0;
		for (int j = 0; j < cols_; ++j) {
			e += this->operator()(i, j) * x[j];
		}
		y[i] = e;
	}

	return y;
}

Matrix &Matrix::operator*=(double lambda)
{
	for (double &x : arr_) {
		x *= lambda;
	}

	return *this;
}

Matrix Matrix::operator/(double lambda) const
{
	Matrix m(rows_, cols_);

	for (std::size_t i = 0; i < arr_.size(); ++i) {
		m.arr_[i] = arr_[i] / lambda;
	}

	return m;
}

Matrix &Matrix::operator/=(double lambda)
{
	for (double &x : arr_) {
		x /= lambda;
	}

	return *this;
}

Matrix Matrix::transpose() const
{
	Matrix m(cols_, rows_);

	for (int i = 0; i < m.rows_; ++i) {
		for (int j = 0; j < m.cols_; ++j) {
			m(i, j) = this->operator()(j, i);
		}
	}

	return m;
}

void Matrix::zero()
{
	std::fill(arr_.begin(), arr_.end(), 0.0);
}

void Matrix::affineTransform(const std::vector<double> &x, const std::vector<double> &b, std::vector<double> &out) const
{
	if (cols_ != (int) x.size() || rows_ != (int) b.size() || rows_ != (int) out.size()) {
		throw std::invalid_argument("dimension mismatch");
	}

	for (int i = 0; i < rows_; ++i) {
		double e = 0.0;
		for (int j = 0; j < cols_; ++j) {
			e += this->operator()(i, j) * x[j];
		}
		out[i] = e + b[i];
	}
}

void Matrix::transposeProduct(const std::vector<double> &x, std::vector<double> &out) const
{
	if (rows_ != (int) x.size()) {
		throw std::invalid_argument("dimension mismatch");
	}

	std::fill(out.begin(), out.end(), 0.0);

	for (int i = 0; i < rows_; ++i) {
		double xi = x[i];
		for (int j = 0; j < cols_; ++j) {
			out[j] += arr_[i * cols_ + j] * xi;
		}
	}
}

void Matrix::addScaled(double lambda, const Matrix &m)
{
	if (m.rows_ != rows_ || m.cols_ != cols_) {
		throw std::invalid_argument("dimension mismatch");
	}

	for (std::size_t i = 0; i < arr_.size(); ++i) {
		arr_[i] += lambda * m.arr_[i];
	}
}

void Matrix::addOuter(const std::vector<double> &a, const std::vector<double> &b)
{
	if ((int) a.size() != rows_ || (int) b.size() != cols_) {
		throw std::invalid_argument("dimension mismatch");
	}

	for (std::size_t i = 0; i < a.size(); ++i) {
		if (a[i] == 0.0) {
			continue;
		}
		for (std::size_t j = 0; j < b.size(); ++j) {
			this->operator()(i, j) +=  a[i] * b[j];
		}
	}
}

Matrix operator*(double lambda, const Matrix &m)
{
	return m * lambda;
}

std::vector<double> operator*(const std::vector<double> &x, const Matrix &m)
{
	if (m.rows_ != (int) x.size()) {
		throw std::invalid_argument("dimension mismatch");
	}

	std::vector<double> y(m.cols_, 0.0);

	for (int i = 0; i < m.rows_; ++i) {
		double xi = x[i];
		for (int j = 0; j < m.cols_; ++j) {
			y[j] += m.arr_[i * m.cols_ + j] * xi;
		}
	}

	return y;
}

std::ostream &operator<<(std::ostream &os, const Matrix &m)
{
	for (int i = 0; i < m.rows_; ++i) {
		os << "|";
		for (int j = 0; j < m.cols_; ++j) {

			os << m(i, j);

			if (j + 1 < m.cols_) {
				os << ", ";	
			}
		}

		os << "|\n";
	}

	return os;
}
