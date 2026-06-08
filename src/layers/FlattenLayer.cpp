#include "layers/FlattenLayer.h"
#include "base/Layer.h"

#include <memory>
#include <stdexcept>

namespace nn {

void FlattenLayerRecipe::validateRecipe() const
{
    LayerRecipe::validateRecipe();

    if (outputShape.dimensions != std::vector<size_t>{inputShape.elementCount()}) {
        throw std::invalid_argument(
            "Flatten output shape must match flattened input shape.");
    }
}

FlattenLayer::FlattenLayer(const FlattenLayerRecipe &newRecipe)
    : Layer(std::make_unique<FlattenLayerRecipe>(newRecipe))
{}

FlattenLayer::~FlattenLayer() = default;

Pattern FlattenLayer::forward(const Pattern &input) const
{
    Pattern output = input;
    output.reshape(getOutputShape());
    return output;
}

} // namespace nn
