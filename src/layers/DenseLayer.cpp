#include "layers/DenseLayer.h"

#include <stdexcept>

namespace
{
Shape denseInputShape(const DenseLayerRecipe &recipe)
{
    return recipe.expectedInputShape.isValid()
               ? recipe.expectedInputShape
               : Shape{recipe.inputSize};
}

Shape denseOutputShape(const DenseLayerRecipe &recipe)
{
    return recipe.expectedOutputShape.isValid()
               ? recipe.expectedOutputShape
               : Shape{recipe.outputSize};
}
}

DenseLayer::DenseLayer(const DenseLayerRecipe &newRecipe)
    : Layer(newRecipe, denseInputShape(newRecipe), denseOutputShape(newRecipe))
{
    if (!newRecipe.isValid()) {
        throw std::invalid_argument("Invalid dense layer recipe");
    }
}

DenseLayer::~DenseLayer() = default;

bool DenseLayer::usesParameters() const
{
    return true;
}

Shape DenseLayer::expectedWeightShape() const
{
    return {getOutputSize(), getInputSize()};
}

Shape DenseLayer::expectedBiasShape() const
{
    return {getOutputSize()};
}

bool DenseLayer::hasBias() const
{
    return !expectedBiasShape().dimensions.empty();
}

bool DenseLayer::hasWeights() const
{
    return !expectedWeightShape().dimensions.empty();
}

const Pattern &DenseLayer::getWeights() const
{
    return weights;
}

const Pattern &DenseLayer::getBiases() const
{
    return biases;
}

LayerParameters DenseLayer::getParameters() const
{
    return {
        .weights = weights,
        .biases = biases,
    };
}

void DenseLayer::setParameters(const LayerParameters &parameters)
{
    setWeights(parameters.weights);
    setBiases(parameters.biases);
}

void DenseLayer::setWeights(const Pattern &newWeights)
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

void DenseLayer::setBiases(const Pattern &newBiases)
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

bool DenseLayer::isInitialized() const
{
    return isInitialized(getParameters());
}

bool DenseLayer::isInitialized(const LayerParameters &parameters) const
{
    return (!hasWeights() || parameters.weights.hasShape(expectedWeightShape()))
           && (!hasBias() || parameters.biases.hasShape(expectedBiasShape()));
}

void DenseLayer::requireInitialized() const
{
    requireInitialized(getParameters());
}

void DenseLayer::requireInitialized(const LayerParameters &parameters) const
{
    if (!isInitialized(parameters)) {
        throw std::runtime_error("Layer weights are not initialized.");
    }
}

Pattern DenseLayer::forward(const Pattern &input) const
{
    Pattern sums = weightedInput(input);
    return activate(sums);
}

Pattern DenseLayer::forward(const Pattern &input,
                            const LayerParameters &parameters) const
{
    Pattern sums = weightedInput(input, parameters);
    return activate(sums);
}

Pattern DenseLayer::weightedInput(const Pattern &input) const
{
    return weightedInput(input, getParameters());
}

Pattern DenseLayer::weightedInput(const Pattern &input,
                                  const LayerParameters &parameters) const
{
    requireInitialized(parameters);

    if (input.empty()) {
        throw std::runtime_error("Input is empty");
    }
    Pattern sums = input.matVec(parameters.weights);
    return hasBias() ? sums + parameters.biases : sums;
}

Pattern DenseLayer::activate(const Pattern &values) const
{
    if (recipe.activation == nullptr) {
        throw std::runtime_error(
            "Activation function is not set for this layer.");
    }

    return values.map(
        [this](Scalar value) { return (*recipe.activation)(value); });
}
