#include "layers/HopfieldLayer.h"
#include <stdexcept>

namespace
{
Shape hopfieldShape(const HopfieldLayerRecipe &recipe)
{
    return recipe.expectedShape.isValid() ? recipe.expectedShape : Shape{recipe.size};
}
}

HopfieldLayer::HopfieldLayer(const HopfieldLayerRecipe &newRecipe)
    : Layer(newRecipe, hopfieldShape(newRecipe), hopfieldShape(newRecipe))
{
    if (!newRecipe.isValid()) {
        throw std::invalid_argument("Invalid hopfield layer recipe");
    }
    ownedParameters = Parameters{};
}

HopfieldLayer::~HopfieldLayer() = default;

bool HopfieldLayer::usesParameters() const
{
    return true;
}

Shape HopfieldLayer::expectedWeightShape() const
{
    return {getOutputSize(), getInputSize()};
}

Shape HopfieldLayer::expectedBiasShape() const
{
    return {};
}

bool HopfieldLayer::hasBias() const
{
    return !expectedBiasShape().dimensions.empty();
}

bool HopfieldLayer::hasWeights() const
{
    return !expectedWeightShape().dimensions.empty();
}

bool HopfieldLayer::acceptsParameters(const Parameters &parameters) const
{
    return (hasWeights() ? parameters.weights.hasShape(expectedWeightShape())
                         : parameters.weights.empty())
           && (hasBias() ? parameters.biases.hasShape(expectedBiasShape())
                         : parameters.biases.empty());
}

void HopfieldLayer::requireValidParameters(const Parameters &parameters) const
{
    if (!acceptsParameters(parameters)) {
        throw std::runtime_error(
            "Layer parameters do not match expected shapes.");
    }
}

Pattern HopfieldLayer::forward(const Pattern &input) const
{
    (void)input;
    throw std::runtime_error("Hopfield layer requires parameters.");
}

Pattern HopfieldLayer::forward(const Pattern &input,
                               const Parameters &parameters) const
{
    return recall(input, parameters);
}

Pattern HopfieldLayer::weightedInput(
    const Pattern &input,
    const Parameters &parameters) const
{
    if (input.empty()) {
        throw std::runtime_error("Input is empty");
    }
    Pattern sums = input.matVec(parameters.weights);
    return hasBias() ? sums + parameters.biases : sums;
}

Pattern HopfieldLayer::recall(const Pattern &input,
                              const Parameters &parameters) const
{
    if (!input.hasShape(getExpectedInputShape())) {
        throw std::runtime_error("Input shape does not match Hopfield layer shape.");
    }

    Pattern state = input;
    Pattern prev_state;
    do {
        prev_state = state;
        auto sum = weightedInput(state, parameters);
        state = activate(sum);
    } while (state != prev_state);
    return state;
}
