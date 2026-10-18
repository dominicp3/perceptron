#pragma once
#include <vector>
#include <ostream>
#include <utility>
#include <initializer_list>

class Matrix
{
public:
	Matrix(int n_rows, int m_cols, float value = 0.0);
	Matrix(std::initializer_list<std::initializer_list<float>> rows);
	Matrix(int n_rows, int m_cols, std::vector<float> &&arr);

	static Matrix identity(int n);
	static Matrix random(int n_rows, int m_cols, float lo=-1.0, float hi=1.0);

	int numRows() const;
	int numCols() const;
	std::pair<int, int> shape() const;

	const float *data() const;

	float operator()(int i, int j) const;
	float& operator()(int i, int j);

	Matrix operator+(const Matrix &other) const;
	Matrix &operator+=(const Matrix &other);

	Matrix operator-(const Matrix &other) const;
	Matrix &operator-=(const Matrix &other);

	Matrix operator*(float lambda) const;
	Matrix operator*(const Matrix &other) const;
	std::vector<float> operator*(const std::vector<float> &x) const;

	Matrix &operator*=(float lambda);

	Matrix operator/(float lambda) const;

	Matrix &operator/=(float lambda);

	bool operator==(const Matrix &other) const;

	bool operator!=(const Matrix &other) const;
	
	Matrix transpose() const;

	void zero();

	/* affine transform performs Mx + b */
	void affineTransform(const std::vector<float> &x, const std::vector<float> &b, std::vector<float> &out) const;

	/* computes W^T x and puts it in out */
	void transposeProduct(const std::vector<float> &x, std::vector<float> &out) const;

	/* computes M += lambda * m */
	void addScaled(float lambda, const Matrix &m);

	/* computes M += a (outer prod) b */
	void addOuter(const std::vector<float> &a, const std::vector<float> &b);

	friend Matrix operator*(float lambda, const Matrix &m);
	friend std::vector<float> operator*(const std::vector<float> &x, const Matrix &m);
	friend std::ostream &operator<<(std::ostream &os, const Matrix &m);

private:
	int rows_, cols_;
	std::vector<float> arr_;
};
