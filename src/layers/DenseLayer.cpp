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

bool DenseLayer::isInitialized(const LayerParameters &parameters) const
{
    return (hasWeights() ? parameters.weights.hasShape(expectedWeightShape())
                         : parameters.weights.empty())
           && (hasBias() ? parameters.biases.hasShape(expectedBiasShape())
                         : parameters.biases.empty());
}

void DenseLayer::requireInitialized(const LayerParameters &parameters) const
{
    if (!isInitialized(parameters)) {
        throw std::runtime_error("Layer weights are not initialized.");
    }
}

Pattern DenseLayer::forward(const Pattern &input) const
{
    (void)input;
    throw std::runtime_error("Dense layer requires parameters.");
}

Pattern DenseLayer::forward(const Pattern &input,
                            const LayerParameters &parameters) const
{
    Pattern sums = weightedInput(input, parameters);
    return activate(sums);
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
