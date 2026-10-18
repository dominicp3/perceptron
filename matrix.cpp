#include "matrix.hpp"
#include <random>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace {
	std::random_device rd;
	std::mt19937_64 gen(rd());
}

Matrix::Matrix(int n_rows, int m_cols, float value):
	rows_{n_rows},
	cols_{m_cols},
	arr_(n_rows * m_cols, value) {}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> rows)
{
	if (rows.size() == 0) {
		throw std::invalid_argument("Matrix cannot be empty");
	}

	std::size_t n_rows = rows.size();
	std::size_t n_cols = rows.begin()->size();

	rows_ = n_rows;
	cols_ = n_cols;
	arr_ = std::vector<float>(n_rows * n_cols);

	std::size_t i = 0;
	for (const auto &row : rows) {

		if (row.size() != n_cols) {
			throw std::invalid_argument("Rows must have equal length");
		}

		std::size_t j = 0;
		for (float entry : row) {
			arr_[i * n_cols + j] = entry;
			++j;
		}

		++i;
	}
}

Matrix::Matrix(int n_rows, int m_cols, std::vector<float> &&arr)
{
	if (n_rows * m_cols != (int) arr.size()) {
		throw std::invalid_argument("arr doesn't match given dimensions");
	}

	rows_ = n_rows;
	cols_ = m_cols;

	arr_ = std::move(arr);
}

Matrix Matrix::identity(int n)
{
	Matrix m(n, n);

	for (int i = 0; i < n; ++i) {
		m.arr_[i * n + i] = 1;
	}

	return m;
}

Matrix Matrix::random(int n_rows, int m_cols, float lo, float hi)
{
	Matrix m(n_rows, m_cols);

	std::uniform_real_distribution<float> dist(lo, hi);

	for (float &x : m.arr_) {
		x = dist(gen);
	}

	return m;
}

int Matrix::numRows() const
{
	return rows_;
}

int Matrix::numCols() const
{
	return cols_;
}

std::pair<int, int> Matrix::shape() const
{
	return {rows_, cols_};
}

const float *Matrix::data() const
{
	return arr_.data();
}

float Matrix::operator()(int i, int j) const
{
	return arr_[i * cols_ + j];
}

float& Matrix::operator()(int i, int j)
{
	return arr_[i * cols_ + j];
}

Matrix Matrix::operator+(const Matrix &other) const
{
	if (rows_ != other.rows_ || cols_ != other.cols_) {
		throw std::invalid_argument("dimension mismatch");
        }
	
	Matrix m(rows_, cols_);
	std::transform(arr_.begin(), arr_.end(), other.arr_.begin(), m.arr_.begin(), std::plus<float>());

	return m;
}

Matrix &Matrix::operator+=(const Matrix &other)
{
	if (rows_ != other.rows_ || cols_ != other.cols_) {
		throw std::invalid_argument("dimension mismatch");
        }

	std::transform(arr_.begin(), arr_.end(), other.arr_.begin(), arr_.begin(), std::plus<float>());

	return *this;
}

Matrix Matrix::operator-(const Matrix &other) const
{
	if (rows_ != other.rows_ || cols_ != other.cols_) {
		throw std::invalid_argument("dimension mismatch");
        }

	Matrix m(rows_, cols_);
	std::transform(arr_.begin(), arr_.end(), other.arr_.begin(), m.arr_.begin(), std::minus<float>());
	return m;
}

Matrix &Matrix::operator-=(const Matrix &other)
{
	if (rows_ != other.rows_ || cols_ != other.cols_) {
		throw std::invalid_argument("dimension mismatch");
        }

	std::transform(arr_.begin(), arr_.end(), other.arr_.begin(), arr_.begin(), std::minus<float>());
	return *this;
}

Matrix Matrix::operator*(float lambda) const
{
	Matrix m(rows_, cols_);
	std::transform(arr_.begin(), arr_.end(), m.arr_.begin(), [lambda](float e) { return lambda * e; });
	return m;
}

Matrix Matrix::operator*(const Matrix &other) const
{
	if (cols_ != other.rows_) {
		throw std::invalid_argument("dimension mismatch");
	}

	Matrix m(rows_, other.cols_);

	for (int i = 0; i < rows_; i++) {
		for (int k = 0; k < cols_; k++) {
			float r = arr_[i * cols_ + k];
			for (int j = 0; j < other.cols_; j++) {
				m.arr_[i * m.cols_ + j] += r * other.arr_[k * other.cols_ + j];
			}
		}
	}

	return m;
}

std::vector<float> Matrix::operator*(const std::vector<float> &x) const
{
	if (cols_ != (int) x.size()) {
		throw std::invalid_argument("dimension mismatch");
	}

	std::vector<float> y(rows_);

	for (int i = 0; i < rows_; ++i) {
		float e = 0.0;
		for (int j = 0; j < cols_; ++j) {
			e += arr_[i * cols_ + j] * x[j];
		}
		y[i] = e;
	}

	return y;
}

Matrix &Matrix::operator*=(float lambda)
{
	std::transform(arr_.begin(), arr_.end(), arr_.begin(), [lambda](float e) { return lambda * e; });
	return *this;
}

Matrix Matrix::operator/(float lambda) const
{
	Matrix m(rows_, cols_);
	std::transform(arr_.begin(), arr_.end(), m.arr_.begin(), [lambda](float e) { return e / lambda; });
	return m;
}

Matrix &Matrix::operator/=(float lambda)
{
	std::transform(arr_.begin(), arr_.end(), arr_.begin(), [lambda](float e) { return e / lambda; });
	return *this;
}

bool Matrix::operator==(const Matrix &other) const
{
	return rows_ == other.rows_ && cols_ == other.cols_ && arr_ == other.arr_;
}

bool Matrix::operator!=(const Matrix &other) const
{
	return rows_ != other.rows_ || cols_ != other.cols_ || arr_ != other.arr_;
}

Matrix Matrix::transpose() const
{
	Matrix m(cols_, rows_);

	for (int i = 0; i < m.rows_; ++i) {
		for (int j = 0; j < m.cols_; ++j) {
			m.arr_[i * m.cols_ + j] = arr_[j * cols_ + i];
		}
	}

	return m;
}

void Matrix::zero()
{
	std::fill(arr_.begin(), arr_.end(), 0.0);
}

void Matrix::affineTransform(const std::vector<float> &x, const std::vector<float> &b, std::vector<float> &out) const
{
	if (cols_ != (int) x.size() || rows_ != (int) b.size() || rows_ != (int) out.size()) {
		throw std::invalid_argument("dimension mismatch");
	}

	for (int i = 0; i < rows_; ++i) {
		float e = 0.0;
		for (int j = 0; j < cols_; ++j) {
			e += arr_[i * cols_ + j] * x[j];
		}
		out[i] = e + b[i];
	}
}

void Matrix::transposeProduct(const std::vector<float> &x, std::vector<float> &out) const
{
	if (rows_ != (int) x.size() || cols_ != (int) out.size()) {
		throw std::invalid_argument("dimension mismatch");
	}

	std::fill(out.begin(), out.end(), 0.0);

	for (int i = 0; i < rows_; ++i) {
		float r = x[i];
		for (int j = 0; j < cols_; ++j) {
			out[j] += arr_[i * cols_ + j] * r;
		}
	}
}

void Matrix::addScaled(float lambda, const Matrix &m)
{
	if (m.rows_ != rows_ || m.cols_ != cols_) {
		throw std::invalid_argument("dimension mismatch");
	}

	for (std::size_t i = 0; i < arr_.size(); ++i) {
		arr_[i] += lambda * m.arr_[i];
	}
}

void Matrix::addOuter(const std::vector<float> &a, const std::vector<float> &b)
{
	if ((int) a.size() != rows_ || (int) b.size() != cols_) {
		throw std::invalid_argument("dimension mismatch");
	}

	for (std::size_t i = 0; i < a.size(); ++i) {
		if (a[i] == 0.0) {
			continue;
		}
		for (std::size_t j = 0; j < b.size(); ++j) {
			arr_[i * cols_ + j] += a[i] * b[j];
		}
	}
}

Matrix operator*(float lambda, const Matrix &m)
{
	return m * lambda;
}

std::vector<float> operator*(const std::vector<float> &x, const Matrix &m)
{
	if (m.rows_ != (int) x.size()) {
		throw std::invalid_argument("dimension mismatch");
	}

	std::vector<float> y(m.cols_, 0.0);

	for (int i = 0; i < m.rows_; ++i) {
		float xi = x[i];
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
