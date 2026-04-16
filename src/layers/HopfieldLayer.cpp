#include "layers/HopfieldLayer.h"

#include <memory>
#include <stdexcept>

HopfieldLayer::HopfieldLayer(const LayerConfig &newConfig)
    : Layer(newConfig)
{}

HopfieldLayer::~HopfieldLayer() = default;

Shape HopfieldLayer::expectedWeightShape() const
{
    return {config.outputSize, config.inputSize};
}

Shape HopfieldLayer::expectedBiasShape() const
{
    return {};
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

    Pattern weightGradients = pattern.outer(pattern);
    weightGradients.setDiagonal(Scalar{});

    weights = weights.zip(weightGradients, [this, learningRate](Scalar weight,
                                                                      Scalar gradient) {
        return config.learningRule->updateWeight(weight, gradient, learningRate);
    });
    weights.setDiagonal(Scalar{});
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
