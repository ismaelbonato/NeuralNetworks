#include "layers/FlattenLayer.h"
#include "base/Layer.h"

#include <memory>
#include <stdexcept>

namespace nn {

Shape FlattenLayerRecipe::getInputShape() const
{
    return inputShape;
}

Shape FlattenLayerRecipe::getOutputShape() const
{
    return {inputShape.elementCount()};
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
