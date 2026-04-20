#include "layers/HopfieldLayer.h"
#include <stdexcept>

namespace
{
Shape hopfieldShape(const HopfieldLayerConfig &config)
{
    return config.expectedShape.isValid() ? config.expectedShape : Shape{config.size};
}
}

HopfieldLayer::HopfieldLayer(const HopfieldLayerConfig &newConfig)
    : TrainableLayer(newConfig, hopfieldShape(newConfig), hopfieldShape(newConfig))
{
    if (!newConfig.isValid()) {
        throw std::invalid_argument("Invalid hopfield layer configuration");
    }
}

HopfieldLayer::~HopfieldLayer() = default;

Shape HopfieldLayer::expectedWeightShape() const
{
    return {getOutputSize(), getInputSize()};
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
    const size_t n = pattern.size();
    if (n != getInputSize() || n != getOutputSize()) {
        throw std::runtime_error("Pattern size does not match Hopfield layer size.");
    }
    if (!pattern.hasShape(getExpectedInputShape())) {
        throw std::runtime_error("Pattern shape does not match Hopfield layer shape.");
    }

    Pattern weightGradients = pattern.outer(pattern);
    weightGradients.setDiagonal(Scalar{});

    weights = weights.zip(weightGradients, [this, learningRate](Scalar weight,
                                                                      Scalar gradient) {
        return trainableConfig.learningRule->updateWeight(weight, gradient, learningRate);
    });
    weights.setDiagonal(Scalar{});
}


Pattern HopfieldLayer::recall(const Pattern &input) const
{
    if (!input.hasShape(getExpectedInputShape())) {
        throw std::runtime_error("Input shape does not match Hopfield layer shape.");
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
