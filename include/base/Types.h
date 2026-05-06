#pragma once

namespace nn {

template<typename T>
class Tensor;

class Layer; // Forward declaration

using Scalar = float;
using Pattern = Tensor<Scalar>;
using Batch = Tensor<Pattern>;

} // namespace nn
