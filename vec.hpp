#pragma once
#include <vector>
#include <ostream>

std::vector<float> &operator+=(std::vector<float> &a, const std::vector<float> &b);
std::vector<float> &addScaled(std::vector<float> &a, const std::vector<float> &b, float lambda);
std::vector<float> &hadamardInplace(std::vector<float> &a, const std::vector<float> &b);
float crossEntropy(const std::vector<float> &z, int k);
std::ostream &operator<<(std::ostream &os, const std::vector<float> &v);
