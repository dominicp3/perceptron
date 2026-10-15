#pragma once
#include <vector>

struct Dataset {
	std::vector<std::vector<double>> X;
	std::vector<int> y;
	int num_classes;
};
