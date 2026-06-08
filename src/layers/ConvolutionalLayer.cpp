#include "layers/ConvolutionalLayer.h"
#include "base/Layer.h"
#include <memory>
#include <stdexcept>

namespace nn {

void ConvolutionalLayerRecipe::validateRecipe() const
{
    LayerRecipe::validateRecipe();

    if (inputShape.dimensions.size() != 2) {
        throw std::invalid_argument(
            "Convolutional input shape must be {channels, length}.");
    }

    if (outputShape.dimensions.size() != 2) {
        throw std::invalid_argument(
            "Convolutional output shape must be {channels, length}.");
    }

    if (kernelSize == 0 || stride == 0) {
        throw std::invalid_argument(
            "Convolutional kernel size and stride must be greater than zero.");
    }

    const size_t inputLength = inputShape.dimensions.at(1);
    const size_t paddedInputLength = inputLength + (2 * padding);
    if (kernelSize > paddedInputLength) {
        throw std::invalid_argument(
            "Convolutional kernel is larger than padded input.");
    }

    const size_t geometryOutputLength = ((paddedInputLength - kernelSize)
                                         / stride)
                                        + 1;
    if (outputShape.dimensions.at(1) != geometryOutputLength) {
        throw std::invalid_argument(
            "Convolutional output shape does not match geometry.");
    }

    if (!activation) {
        throw std::invalid_argument(
            "Convolutional layer requires an activation.");
    }
}

ConvolutionalLayer::ConvolutionalLayer(const ConvolutionalLayerRecipe &newRecipe)
    : Layer(std::make_unique<ConvolutionalLayerRecipe>(newRecipe))
{}

ConvolutionalLayer::~ConvolutionalLayer() = default;

Shape ConvolutionalLayer::expectedWeightShape() const
{
    return {outputChannelsFromShape(),
            inputChannelsFromShape(),
            getKernelSize()};
}

Shape ConvolutionalLayer::expectedBiasShape() const
{
    return {outputChannelsFromShape()};
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
    for (size_t outputChannel = 0; outputChannel < outputChannelsFromShape();
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

size_t ConvolutionalLayer::inputChannelsFromShape() const
{
    return getInputShape().dimensions.at(0);
}

size_t ConvolutionalLayer::inputLengthFromShape() const
{
    return getInputShape().dimensions.at(1);
}

size_t ConvolutionalLayer::outputChannelsFromShape() const
{
    return getOutputShape().dimensions.at(0);
}

size_t ConvolutionalLayer::outputLengthFromShape() const
{
    return getOutputShape().dimensions.at(1);
}

size_t ConvolutionalLayer::outputLengthFromGeometry() const
{
    const auto &recipe = recipeConfig();
    return ((inputLengthFromShape() + (2 * recipe.padding) - recipe.kernelSize)
            / recipe.stride)
           + 1;
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

LayerSnapshot ConvolutionalLayer::snapshot() const
{
    auto snapshot = Layer::snapshot();

    snapshot.fields.push_back({"kernelSize", getKernelSize()});
    snapshot.fields.push_back({"stride", getStride()});
    snapshot.fields.push_back({"padding", getPadding()});
    return snapshot;
}

} // namespace nn
