#pragma once
#include <vector>
#include <ostream>
#include <utility>
#include <initializer_list>

class Matrix
{
public:
	Matrix(int n_rows, int m_cols, double value = 0.0);
	Matrix(std::initializer_list<std::initializer_list<double>> rows);

	static Matrix identity(int n);
	static Matrix random(int n_rows, int m_cols, double lo=-1.0, double hi=1.0);

	std::pair<int, int> shape() const;

	double operator()(int i, int j) const;
	double& operator()(int i, int j);

	Matrix operator+(const Matrix &other) const;
	Matrix &operator+=(const Matrix &other);

	Matrix operator-(const Matrix &other) const;
	Matrix &operator-=(const Matrix &other);

	Matrix operator*(double lambda) const;
	Matrix operator*(const Matrix &other) const;
	std::vector<double> operator*(const std::vector<double> &x) const;

	Matrix &operator*=(double lambda);

	Matrix operator/(double lambda) const;

	Matrix &operator/=(double lambda);
	
	Matrix transpose() const;

	void zero();

	/* affine transform performs Mx + b */
	void affineTransform(const std::vector<double> &x, const std::vector<double> &b, std::vector<double> &out) const;

	/* computes W^T x and puts it in out */
	void transposeProduct(const std::vector<double> &x, std::vector<double> &out) const;

	/* computes M += lambda * m */
	void addScaled(double lambda, const Matrix &m);

	/* computes M += a (outer prod) b */
	void addOuter(const std::vector<double> &a, const std::vector<double> &b);

	friend Matrix operator*(double lambda, const Matrix &m);
	friend std::vector<double> operator*(const std::vector<double> &x, const Matrix &m);
	friend std::ostream &operator<<(std::ostream &os, const Matrix &m);

private:
	int rows_, cols_;
	std::vector<double> arr_;
};
