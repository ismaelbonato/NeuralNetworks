#include "layers/DenseLayer.h"

#include <memory>

DenseLayer::DenseLayer(const LayerConfig &newConfig)
    : Layer(newConfig)
{}

DenseLayer::~DenseLayer() = default;

std::shared_ptr<Layer> DenseLayer::clone() const
{
    auto cloned = std::make_shared<DenseLayer>(config);
    cloned->weights = weights;
    cloned->biases = biases;
    return cloned;
}

