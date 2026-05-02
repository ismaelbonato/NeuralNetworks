#include "layers/ConvolutionalLayer.h"
#include <stdexcept>

namespace {

size_t convolutionalOutputLength(const ConvolutionalLayerRecipe &recipe)
{
    return ((recipe.inputLength + (2 * recipe.padding) - recipe.kernelSize)
            / recipe.stride)
           + 1;
}

Shape convolutionalInputShape(const ConvolutionalLayerRecipe &recipe)
{
    return {recipe.inputChannels, recipe.inputLength};
}

Shape convolutionalOutputShape(const ConvolutionalLayerRecipe &recipe)
{
    return {recipe.outputChannels, convolutionalOutputLength(recipe)};
}

} // namespace

ConvolutionalLayer::ConvolutionalLayer(const ConvolutionalLayerRecipe &newRecipe)
    : Layer(newRecipe,
            convolutionalInputShape(newRecipe),
            convolutionalOutputShape(newRecipe))
    , convolutionalRecipe(newRecipe)
{
    if (!newRecipe.isValid()) {
        throw std::invalid_argument(
            "Invalid convolutional layer recipe");
    }
}

ConvolutionalLayer::~ConvolutionalLayer() = default;

bool ConvolutionalLayer::usesParameters() const
{
    return true;
}

Shape ConvolutionalLayer::expectedWeightShape() const
{
    return {convolutionalRecipe.outputChannels,
            convolutionalRecipe.inputChannels,
            convolutionalRecipe.kernelSize};
}

Shape ConvolutionalLayer::expectedBiasShape() const
{
    return {convolutionalRecipe.outputChannels};
}

bool ConvolutionalLayer::hasBias() const
{
    return !expectedBiasShape().dimensions.empty();
}

bool ConvolutionalLayer::hasWeights() const
{
    return !expectedWeightShape().dimensions.empty();
}

const Pattern &ConvolutionalLayer::getWeights() const
{
    return weights;
}

const Pattern &ConvolutionalLayer::getBiases() const
{
    return biases;
}

LayerParameters ConvolutionalLayer::getParameters() const
{
    return {
        .weights = weights,
        .biases = biases,
    };
}

void ConvolutionalLayer::setParameters(const LayerParameters &parameters)
{
    setWeights(parameters.weights);
    setBiases(parameters.biases);
}

void ConvolutionalLayer::setWeights(const Pattern &newWeights)
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

void ConvolutionalLayer::setBiases(const Pattern &newBiases)
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

bool ConvolutionalLayer::isInitialized() const
{
    return isInitialized(getParameters());
}

bool ConvolutionalLayer::isInitialized(
    const LayerParameters &parameters) const
{
    return (hasWeights() ? parameters.weights.hasShape(expectedWeightShape())
                         : parameters.weights.empty())
           && (hasBias() ? parameters.biases.hasShape(expectedBiasShape())
                         : parameters.biases.empty());
}

void ConvolutionalLayer::requireInitialized() const
{
    requireInitialized(getParameters());
}

void ConvolutionalLayer::requireInitialized(
    const LayerParameters &parameters) const
{
    if (!isInitialized(parameters)) {
        throw std::runtime_error("Layer weights are not initialized.");
    }
}

Pattern ConvolutionalLayer::forward(const Pattern &input) const
{
    Pattern sums = weightedInput(input);
    return activate(sums);
}

Pattern ConvolutionalLayer::forward(
    const Pattern &input,
    const LayerParameters &parameters) const
{
    Pattern sums = weightedInput(input, parameters);
    return activate(sums);
}

Pattern ConvolutionalLayer::weightedInput(const Pattern &input) const
{
    return weightedInput(input, getParameters());
}

Pattern ConvolutionalLayer::weightedInput(
    const Pattern &input,
    const LayerParameters &parameters) const
{
    requireInitialized(parameters);

    if (input.empty()) {
        throw std::runtime_error("Input is empty");
    }

    Pattern result = input.conv1D(parameters.weights,
                                  convolutionalRecipe.stride,
                                  convolutionalRecipe.padding);
    if (hasBias()) {
        for (size_t outputChannel = 0;
             outputChannel < convolutionalRecipe.outputChannels;
             ++outputChannel) {
            for (size_t outputIndex = 0; outputIndex < result.shape().at(1);
                 ++outputIndex) {
                result.at({outputChannel, outputIndex})
                    += parameters.biases.at(outputChannel);
            }
        }
    }

    return result;
}

Pattern ConvolutionalLayer::activate(const Pattern &values) const
{
    if (recipe.activation == nullptr) {
        throw std::runtime_error(
            "Activation function is not set for this layer.");
    }

    return values.map(
        [this](Scalar value) { return (*recipe.activation)(value); });
}

const ConvolutionalLayerRecipe &ConvolutionalLayer::getConvolutionalRecipe() const
{
    return convolutionalRecipe;
}
