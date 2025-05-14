#pragma once

#include "matrix.hpp"
#include <vector>
#include <string>
#include <utility>

std::pair<std::vector<Matrix<float>>, std::vector<Matrix<float>>>
loadIrisDataset(const std::string& path);

std::pair<std::vector<Matrix<float>>, std::vector<Matrix<float>>>
generateXORDataset();
