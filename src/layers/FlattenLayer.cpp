#include "layers/FlattenLayer.h"

#include <stdexcept>

namespace nn {

FlattenLayer::FlattenLayer(const FlattenLayerRecipe &newRecipe)
    : Layer(newRecipe,
            newRecipe.expectedInputShape,
            newRecipe.expectedOutputShape())
{
    if (!newRecipe.isValid()) {
        throw std::invalid_argument("Invalid flatten layer recipe");
    }
}

FlattenLayer::~FlattenLayer() = default;

Pattern FlattenLayer::forward(const Pattern &input) const
{
    Pattern output = input;
    output.reshape(getExpectedOutputShape());
    return output;
}

} // namespace nn
