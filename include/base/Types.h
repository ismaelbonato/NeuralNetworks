#pragma once
#include <vector>
#include <memory>
#include <utility>
#include <iostream>
#include <stdexcept>
#include <cstddef> // for size_t

class Layer; // Forward declaration

using Scalar = float;
using Pattern = std::vector<Scalar>;
using Patterns = std::vector<Pattern>;
using Matrix = std::vector<std::vector<Scalar>>;
using MatrixPattern = std::vector<Pattern>;

using PatternPair = std::pair<Pattern, Pattern>;

using Layers = std::vector<std::unique_ptr<Layer>>;
