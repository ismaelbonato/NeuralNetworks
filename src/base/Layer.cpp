#include "base/Layer.h"

#include <random>
#include <stdexcept>

bool LayerConfig::isValid() const
{
    return learningRule && activation && inputSize > 0 && outputSize > 0;
}

Layer::Layer(const LayerConfig &newConfig)
    : config(newConfig)
{
    if (!config.isValid()) {
        throw std::invalid_argument("Invalid layer configuration");
    }
}

Layer::~Layer() = default;

size_t Layer::getInputSize() const
{
    return config.inputSize;
}

size_t Layer::getOutputSize() const
{
    return config.outputSize;
}

void Layer::updateWeights(const Pattern &prev_activations,
                          const Pattern &layerDelta,
                          Scalar learningRate)
{
    if (prev_activations.size() != config.inputSize) {
        throw std::runtime_error("Previous activation size does not match layer input size.");
    }
    if (layerDelta.size() != config.outputSize) {
        throw std::runtime_error("Layer delta size does not match layer output size.");
    }

    for (size_t i = 0; i < config.outputSize; ++i) {
        for (size_t j = 0; j < config.inputSize; ++j) {
            Scalar gradient = layerDelta[i] * prev_activations[j];
            weights[i][j] = config.learningRule->updateWeight(weights[i][j],
                                                              gradient,
                                                              learningRate);
        }
    }

    if (config.useBias) {
        for (size_t i = 0; i < config.outputSize; ++i) {
            biases[i] = config.learningRule->updateWeight(biases[i],
                                                          layerDelta[i],
                                                          learningRate);
        }
    }
}

Pattern Layer::infer(const Pattern &input) const
{
    if (input.size() != config.inputSize) {
        throw std::runtime_error("Input size does not match layer input size.");
    }
    Pattern sums = weightedSum(input);
    return activate(sums);
}

Pattern Layer::weightedSum(const Pattern &input) const
{
    if (input.empty()) {
        throw std::runtime_error("Input is empty");
    }

    Pattern sums = weights.matVecMul(input);
    return config.useBias ? sums + biases : sums;
}

Pattern Layer::activationDerivatives(const Pattern &values) const
{
    Pattern result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = (*config.activation).derivative(values[i]);
    }
    return result;
}

Pattern Layer::activate(const Pattern &values) const
{
    if (config.learningRule == nullptr) {
        throw std::runtime_error("Learning rule is not set for this layer.");
    }
    if (config.activation == nullptr) {
        throw std::runtime_error("Activation function is not set for this layer.");
    }

    Pattern result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = (*config.activation)(values[i]);
    }
    return result;
}

Pattern Layer::backwardPass(const Pattern &layerDelta, const Pattern &preActivation)
{
    if (weights.empty() || weights.size() != config.outputSize) {
        throw std::runtime_error("Weights are not initialized or size mismatch.");
    }
    return weights.matVecTransMul(layerDelta) * activationDerivatives(preActivation);
}

void Layer::naturalUpdateWeights(const Layer &l)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Scalar> dis(-0.01f, 0.01f);

    for (size_t i = 0; i < config.outputSize; ++i) {
        for (size_t j = 0; j < config.inputSize; ++j) {
            weights[i][j] = l.weights[i][j] + dis(gen);
        }
    }

    for (size_t i = 0; i < config.outputSize; ++i) {
        biases[i] = l.biases[i] + dis(gen);
    }
}
