#include "layers/ConvolutionalLayer.h"
#include "base/Layer.h"
#include <memory>
#include <stdexcept>

namespace nn {

namespace {
size_t convolutionalOutputLength(const ConvolutionalLayerRecipe &recipe)
{
    return ((recipe.inputLength + (2 * recipe.padding) - recipe.kernelSize)
            / recipe.stride)
           + 1;
}

} // namespace

Shape ConvolutionalLayerRecipe::getInputShape() const
{
    return {inputChannels, inputLength};
}

Shape ConvolutionalLayerRecipe::getOutputShape() const
{
    return {outputChannels, convolutionalOutputLength(*this)};
}

void ConvolutionalLayerRecipe::validateRecipe() const
{
    if (inputChannels == 0 || inputLength == 0 || outputChannels == 0
        || kernelSize == 0 || stride == 0) {
        throw std::invalid_argument(
            "Convolutional layer dimensions must be greater than zero.");
    }

    const size_t paddedInputLength = inputLength + (2 * padding);
    if (kernelSize > paddedInputLength) {
        throw std::invalid_argument(
            "Convolutional kernel is larger than padded input.");
    }

    if (!activation) {
        throw std::invalid_argument(
            "Convolutional layer requires an activation.");
    }

    LayerRecipe::validateRecipe();
}

ConvolutionalLayer::ConvolutionalLayer(const ConvolutionalLayerRecipe &newRecipe)
    : Layer(std::make_unique<ConvolutionalLayerRecipe>(newRecipe))
{}

ConvolutionalLayer::~ConvolutionalLayer() = default;

Shape ConvolutionalLayer::expectedWeightShape() const
{
    const auto &recipe = recipeConfig();

    return {recipe.outputChannels, recipe.inputChannels, recipe.kernelSize};
}

Shape ConvolutionalLayer::expectedBiasShape() const
{
    const auto &recipe = recipeConfig();
    return {recipe.outputChannels};
}

Pattern ConvolutionalLayer::weightedInput(const Pattern &input,
                                          const Parameters &parameters) const
{
    if (input.empty()) {
        throw std::runtime_error("Input is empty");
    }
    const auto &recipe = recipeConfig();

    Pattern result = input.conv1D(parameters.weights,
                                  recipe.stride,
                                  recipe.padding);
    for (size_t outputChannel = 0; outputChannel < recipe.outputChannels;
         ++outputChannel) {
        for (size_t outputIndex = 0; outputIndex < result.shape().at(1);
             ++outputIndex) {
            result.at({outputChannel, outputIndex}) += parameters.biases.at(
                outputChannel);
        }
    }

    return result;
}

const ConvolutionalLayerRecipe &ConvolutionalLayer::recipeConfig() const
{
    return static_cast<const ConvolutionalLayerRecipe &>(*recipe);
}

size_t ConvolutionalLayer::getInputChannels() const
{
    return recipeConfig().inputChannels;
}

size_t ConvolutionalLayer::getInputLength() const
{
    return recipeConfig().inputLength;
}

size_t ConvolutionalLayer::getOutputChannels() const
{
    return recipeConfig().outputChannels;
}

size_t ConvolutionalLayer::getKernelSize() const
{
    return recipeConfig().kernelSize;
}

size_t ConvolutionalLayer::getStride() const
{
    return recipeConfig().stride;
}

size_t ConvolutionalLayer::getPadding() const
{
    return recipeConfig().padding;
}

} // namespace nn
