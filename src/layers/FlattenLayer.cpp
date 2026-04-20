#include "layers/FlattenLayer.h"

#include <stdexcept>

FlattenLayer::FlattenLayer(const FlattenLayerConfig &newConfig)
    : Layer(newConfig,
            newConfig.expectedInputShape,
            newConfig.expectedOutputShape())
{
    if (!newConfig.isValid()) {
        throw std::invalid_argument("Invalid flatten layer configuration");
    }
}

FlattenLayer::~FlattenLayer() = default;

Pattern FlattenLayer::infer(const Pattern &input) const
{
    requireInputShape(input);

    Pattern output = input;
    output.reshape(getExpectedOutputShape());
    return output;
}

Pattern FlattenLayer::backwardPass(const Pattern &layerDelta,
                                   const Pattern &preActivation) const
{
    if (layerDelta.size() != preActivation.size()) {
        throw std::runtime_error("Flatten layer delta size does not match previous activation size.");
    }

    Pattern previousDelta = layerDelta;
    previousDelta.reshape(Shape(preActivation.shape()));
    return previousDelta;
}
