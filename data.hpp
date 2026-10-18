#pragma once
#include <vector>

struct Dataset {
	std::vector<std::vector<float>> X;
	std::vector<int> y;
	int num_classes;
};
