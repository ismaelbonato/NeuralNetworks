#include "layers/DenseLayer.h"

#include <memory>

DenseLayer::DenseLayer(const LayerConfig &newConfig)
    : Layer(newConfig)
{}

DenseLayer::~DenseLayer() = default;

Shape DenseLayer::expectedWeightShape() const
{
    return {config.outputSize, config.inputSize};
}

Shape DenseLayer::expectedBiasShape() const
{
    return {config.outputSize};
}

