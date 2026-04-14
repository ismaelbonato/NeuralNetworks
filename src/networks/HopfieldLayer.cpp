#include "layers/HopfieldLayer.h"

#include <memory>
#include <stdexcept>

HopfieldLayer::HopfieldLayer(const LayerConfig &newConfig)
    : Layer(newConfig)
{}

HopfieldLayer::~HopfieldLayer() = default;

std::shared_ptr<Layer> HopfieldLayer::clone() const
{
    return std::make_shared<HopfieldLayer>(config);
}

Pattern HopfieldLayer::infer(const Pattern &input) const
{
    return recall(input);
}

void HopfieldLayer::updateWeights(const Pattern &pattern,
                                  const Pattern &,
                                  Scalar learningRate)
{
    size_t n = pattern.size();
    if (n != config.inputSize || n != config.outputSize) {
        throw std::runtime_error("Pattern size does not match Hopfield layer size.");
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i != j) {
                weights[i][j] = config.learningRule->updateWeight(weights[i][j],
                                                                  pattern[i] * pattern[j],
                                                                  learningRate);
            }
        }
    }
}

void HopfieldLayer::initWeights(Scalar value)
{
    if (weights.empty()) {
        weights.resize(config.outputSize, Pattern(config.inputSize, value));
    }

    if (biases.empty()) {
        biases.resize(config.inputSize, value);
    }
}

Pattern HopfieldLayer::recall(const Pattern &input) const
{
    if (input.size() != config.inputSize) {
        throw std::runtime_error("Input size does not match Hopfield layer size.");
    }

    Pattern state = input;
    Pattern prev_state;
    do {
        prev_state = state;
        auto sum = weightedSum(state);
        state = activate(sum);
    } while (state != prev_state);
    return state;
}
