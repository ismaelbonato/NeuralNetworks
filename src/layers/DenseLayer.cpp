#include "layers/DenseLayer.h"
#include "base/Layer.h"

#include <memory>
#include <stdexcept>

namespace nn {

void DenseLayerRecipe::validateRecipe() const
{
    LayerRecipe::validateRecipe();

    if (!activation) {
        throw std::invalid_argument("Dense layer requires an activation.");
    }
}

DenseLayer::DenseLayer(const DenseLayerRecipe &newRecipe)
    : Layer(std::make_unique<DenseLayerRecipe>(newRecipe))
{}

DenseLayer::~DenseLayer() = default;

Shape DenseLayer::expectedWeightShape() const
{
    return {getOutputShape().elementCount(), getInputShape().elementCount()};
}

Shape DenseLayer::expectedBiasShape() const
{
    return {getOutputShape().elementCount()};
}

Pattern DenseLayer::weightedInput(const Pattern &input,
                                  const Parameters &parameters) const
{
    if (input.empty()) {
        throw std::runtime_error("Input is empty");
    }
    Pattern sums = input.matVec(parameters.weights);
    return sums + parameters.biases;
}

} // namespace nn
