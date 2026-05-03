#include "base/Layer.h"

#include <stdexcept>

namespace {
bool isSizeCompatibleWithShape(const size_t size, const Shape &shape)
{
    return size == 0 || shape.empty()
           || (shape.isValid() && shape.elementCount() == size);
}

} // namespace

bool ConvolutionalLayerRecipe::isValid() const
{
    const size_t paddedInputLength = inputLength + (2 * padding);

    return activation && inputChannels > 0 && inputLength > 0
           && outputChannels > 0 && kernelSize > 0 && stride > 0
           && kernelSize <= paddedInputLength;
}

bool DenseLayerRecipe::isValid() const
{
    return activation && (inputSize > 0 || expectedInputShape.isValid())
           && (outputSize > 0 || expectedOutputShape.isValid())
           && isSizeCompatibleWithShape(inputSize, expectedInputShape)
           && isSizeCompatibleWithShape(outputSize, expectedOutputShape);
}

bool HopfieldLayerRecipe::isValid() const
{
    return activation && (size > 0 || expectedShape.isValid())
           && isSizeCompatibleWithShape(size, expectedShape);
}

bool FlattenLayerRecipe::isValid() const
{
    return expectedInputShape.isValid();
}

Shape FlattenLayerRecipe::expectedOutputShape() const
{
    return {expectedInputShape.elementCount()};
}

Layer::Layer(const LayerRecipe &newRecipe,
             const Shape &newExpectedInput,
             const Shape &newExpectedOutput)
    : recipe(newRecipe)
    , expectedInput(newExpectedInput)
    , expectedOutput(newExpectedOutput)
{
    if (!expectedInput.isValid() || !expectedOutput.isValid()) {
        throw std::invalid_argument("Invalid layer recipe");
    }
}

Layer::~Layer() = default;

size_t Layer::getInputSize() const
{
    return expectedInput.elementCount();
}

size_t Layer::getOutputSize() const
{
    return expectedOutput.elementCount();
}

const Shape &Layer::getExpectedInputShape() const
{
    return expectedInput;
}

const Shape &Layer::getExpectedOutputShape() const
{
    return expectedOutput;
}

const Shape &Layer::getInputShape() const
{
    return getExpectedInputShape();
}

const Shape &Layer::getOutputShape() const
{
    return getExpectedOutputShape();
}

const std::shared_ptr<ActivationFunction<Scalar>> &Layer::getActivation() const
{
    return recipe.activation;
}

bool Layer::usesParameters() const
{
    return false;
}

Shape Layer::expectedWeightShape() const
{
    return {};
}

Shape Layer::expectedBiasShape() const
{
    return {};
}

bool Layer::isInitialized(const Parameters &parameters) const
{
    return parameters.weights.empty() && parameters.biases.empty();
}

void Layer::requireInitialized(const Parameters &parameters) const
{
    if (!isInitialized(parameters)) {
        throw std::runtime_error("Layer does not use parameters.");
    }
}

void Layer::requireInputShape(const Pattern &input) const
{
    if (!input.hasShape(expectedInput)) {
        throw std::runtime_error(
            "Input shape does not match layer input shape.");
    }
}

Pattern Layer::infer(const Pattern &input) const
{
    requireInputShape(input);
    return forward(input);
}

Pattern Layer::infer(const Pattern &input, const Parameters &parameters) const
{
    requireInputShape(input);
    return forward(input, parameters);
}

Pattern Layer::forward(const Pattern &input, const Parameters &parameters) const
{
    (void) parameters;
    return forward(input);
}
