#pragma once
#include <vector>
#include <ostream>

std::vector<double> &operator+=(std::vector<double> &a, const std::vector<double> &b);
std::vector<double> &addScaled(std::vector<double> &a, const std::vector<double> &b, double lambda);
std::vector<double> &hadamardInplace(std::vector<double> &a, const std::vector<double> &b);
double crossEntropy(const std::vector<double> &z, int k);
std::ostream &operator<<(std::ostream &os, const std::vector<double> &v);
