#pragma once
#include <vector>

template<typename T>
class Tensor;

class Layer; // Forward declaration
class Skill; // Forward declaration

using Scalar = float;
using Pattern = Tensor<Scalar>;
using Batch = Tensor<Pattern>;

using Skills = std::vector<Skill>;
