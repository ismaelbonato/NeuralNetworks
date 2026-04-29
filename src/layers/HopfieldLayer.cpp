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

bool HopfieldLayer::hasBias() const
{
    return !expectedBiasShape().dimensions.empty();
}

bool HopfieldLayer::hasWeights() const
{
    return !expectedWeightShape().dimensions.empty();
}

const Pattern &HopfieldLayer::getWeights() const
{
    return weights;
}

const Pattern &HopfieldLayer::getBiases() const
{
    return biases;
}

LayerParameters HopfieldLayer::getParameters() const
{
    return {
        .weights = weights,
        .biases = biases,
    };
}

void HopfieldLayer::setParameters(const LayerParameters &parameters)
{
    setWeights(parameters.weights);
    setBiases(parameters.biases);
}

void HopfieldLayer::setWeights(const Pattern &newWeights)
{
    if (!hasWeights()) {
        if (!newWeights.empty()) {
            throw std::runtime_error("Layer does not use weights.");
        }

        weights = newWeights;
        return;
    }

    if (!newWeights.hasShape(expectedWeightShape())) {
        throw std::runtime_error(
            "Layer weights shape does not match layer recipe.");
    }

    weights = newWeights;
}

void HopfieldLayer::setBiases(const Pattern &newBiases)
{
    if (!hasBias()) {
        if (!newBiases.empty()) {
            throw std::runtime_error("Layer does not use bias.");
        }

        biases = newBiases;
        return;
    }

    if (!newBiases.hasShape(expectedBiasShape())) {
        throw std::runtime_error(
            "Layer bias size does not match layer output size.");
    }

    biases = newBiases;
}

bool HopfieldLayer::isInitialized() const
{
    return (!hasWeights() || weights.hasShape(expectedWeightShape()))
           && (!hasBias() || biases.hasShape(expectedBiasShape()));
}

void HopfieldLayer::requireInitialized() const
{
    if (!isInitialized()) {
        throw std::runtime_error("Layer weights are not initialized.");
    }
}

Pattern HopfieldLayer::forward(const Pattern &input) const
{
    return recall(input);
}

Pattern HopfieldLayer::weightedInput(const Pattern &input) const
{
    requireInitialized();

    if (input.empty()) {
        throw std::runtime_error("Input is empty");
    }
    Pattern sums = input.matVec(weights);
    return hasBias() ? sums + biases : sums;
}

Pattern HopfieldLayer::activate(const Pattern &values) const
{
    if (recipe.activation == nullptr) {
        throw std::runtime_error(
            "Activation function is not set for this layer.");
    }

    return values.map(
        [this](Scalar value) { return (*recipe.activation)(value); });
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
        auto sum = weightedInput(state);
        state = activate(sum);
    } while (state != prev_state);
    return state;
}
