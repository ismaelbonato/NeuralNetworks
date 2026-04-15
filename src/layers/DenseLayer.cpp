#include "layers/DenseLayer.h"

#include <memory>
#include <random>

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

void DenseLayer::initWeights(Scalar value)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Scalar> dis(-1.0, 1.0);

    if (weights.empty()) {
        weights = Pattern::matrix(config.outputSize, config.inputSize, value);

        if (config.initWeights) {
            weights.generate([&dis, &gen]() { return dis(gen); });
        }
    }

    if (config.useBias && biases.empty()) {
        biases = Pattern::vector(config.outputSize, config.biasInit);

        if (config.initWeights) {
            biases.generate([&dis, &gen]() { return dis(gen); });
        }
    }
}
