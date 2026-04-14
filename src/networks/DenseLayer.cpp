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
        weights.resize(config.outputSize, Pattern(config.inputSize, value));

        if (config.initWeights) {
            for (auto &row : weights) {
                for (auto &w : row) {
                    w = dis(gen);
                }
            }
        }
    }

    if (config.useBias && biases.empty()) {
        biases.resize(config.outputSize, config.biasInit);

        if (config.initWeights) {
            for (auto &b : biases) {
                b = dis(gen);
            }
        }
    }
}
