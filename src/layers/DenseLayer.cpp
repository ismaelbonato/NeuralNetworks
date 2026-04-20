#include "layers/DenseLayer.h"

#include <stdexcept>

namespace
{
Shape denseInputShape(const DenseLayerConfig &config)
{
    return config.expectedInputShape.isValid()
               ? config.expectedInputShape
               : Shape{config.inputSize};
}

Shape denseOutputShape(const DenseLayerConfig &config)
{
    return config.expectedOutputShape.isValid()
               ? config.expectedOutputShape
               : Shape{config.outputSize};
}
}

DenseLayer::DenseLayer(const DenseLayerConfig &newConfig)
    : TrainableLayer(newConfig, denseInputShape(newConfig), denseOutputShape(newConfig))
{
    if (!newConfig.isValid()) {
        throw std::invalid_argument("Invalid dense layer configuration");
    }
}

DenseLayer::~DenseLayer() = default;

Shape DenseLayer::expectedWeightShape() const
{
    return {getOutputSize(), getInputSize()};
}

Shape DenseLayer::expectedBiasShape() const
{
    return {getOutputSize()};
}
