#pragma once
#include <vector>
#include <memory>

template<typename T>
class Tensor;

class Layer; // Forward declaration

using Scalar = float;
using Pattern = Tensor<Scalar>;
using Batch = Tensor<Pattern>;

using Layers = std::vector<std::unique_ptr<Layer>>;
