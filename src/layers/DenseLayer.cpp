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
    ownedParameters = Parameters{};
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

bool DenseLayer::acceptsParameters(const Parameters &parameters) const
{
    return (hasWeights() ? parameters.weights.hasShape(expectedWeightShape())
                         : parameters.weights.empty())
           && (hasBias() ? parameters.biases.hasShape(expectedBiasShape())
                         : parameters.biases.empty());
}

void DenseLayer::requireValidParameters(const Parameters &parameters) const
{
    if (!acceptsParameters(parameters)) {
        throw std::runtime_error(
            "Layer parameters do not match expected shapes.");
    }
}

Pattern DenseLayer::weightedInput(const Pattern &input,
                                  const Parameters &parameters) const
{
    if (input.empty()) {
        throw std::runtime_error("Input is empty");
    }
    Pattern sums = input.matVec(parameters.weights);
    return hasBias() ? sums + parameters.biases : sums;
}
