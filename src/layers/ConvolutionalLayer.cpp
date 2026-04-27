#include "layers/ConvolutionalLayer.h"
#include "base/Layer.h"
#include <stdexcept>

namespace {

size_t convolutionalOutputLength(const ConvolutionalLayerConfig &config)
{
    return ((config.inputLength + (2 * config.padding) - config.kernelSize)
            / config.stride)
           + 1;
}

Shape convolutionalInputShape(const ConvolutionalLayerConfig &config)
{
    return {config.inputChannels, config.inputLength};
}

Shape convolutionalOutputShape(const ConvolutionalLayerConfig &config)
{
    return {config.outputChannels, convolutionalOutputLength(config)};
}

} // namespace

ConvolutionalLayer::ConvolutionalLayer(const ConvolutionalLayerConfig &newConfig)
    : TrainableLayer(newConfig,
                     convolutionalInputShape(newConfig),
                     convolutionalOutputShape(newConfig))
    , convolutionalConfig(newConfig)
{
    if (!newConfig.isValid()) {
        throw std::invalid_argument(
            "Invalid convolutional layer configuration");
    }
}

ConvolutionalLayer::~ConvolutionalLayer() = default;

Shape ConvolutionalLayer::expectedWeightShape() const
{
    return {convolutionalConfig.outputChannels,
            convolutionalConfig.inputChannels,
            convolutionalConfig.kernelSize};
}

Shape ConvolutionalLayer::expectedBiasShape() const
{
    return {convolutionalConfig.outputChannels};
}

Pattern ConvolutionalLayer::preActivation(const Pattern &input) const
{
    requireInitialized();

    if (input.empty()) {
        throw std::runtime_error("Input is empty");
    }

    Pattern result = input.conv1D(weights,
                                  convolutionalConfig.stride,
                                  convolutionalConfig.padding);
    if (hasBias()) {
        for (size_t outputChannel = 0;
             outputChannel < convolutionalConfig.outputChannels;
             ++outputChannel) {
            for (size_t outputIndex = 0; outputIndex < result.shape().at(1);
                 ++outputIndex) {
                result.at({outputChannel, outputIndex}) += biases.at(
                    outputChannel);
            }
        }
    }

    return result;
}

Pattern ConvolutionalLayer::backwardPass(const Pattern &layerDelta,
                                         const Pattern &layerInput) const
{
    requireInitialized();

    if (!layerDelta.hasShape(expectedOutput)) {
        throw std::runtime_error("Layer delta shape does not match "
                                 "convolutional layer output shape.");
    }
    if (!layerInput.hasShape(expectedInput)) {
        throw std::runtime_error("Layer input shape does not match "
                                 "convolutional layer input shape.");
    }
    Pattern inputDelta = Pattern::withShape(expectedInput, Scalar{0});

    for (size_t outputChannel = 0;
         outputChannel < convolutionalConfig.outputChannels;
         ++outputChannel) {
        for (size_t outputIndex = 0; outputIndex < layerDelta.shape().at(1);
             ++outputIndex) {
            for (size_t inputChannel = 0;
                 inputChannel < convolutionalConfig.inputChannels;
                 ++inputChannel) {
                for (size_t kernelIndex = 0;
                     kernelIndex < convolutionalConfig.kernelSize;
                     ++kernelIndex) {
                    const size_t paddedInputIndex
                        = (outputIndex * convolutionalConfig.stride)
                          + kernelIndex;

                    if (paddedInputIndex < convolutionalConfig.padding) {
                        continue;
                    }

                    const size_t inputIndex = paddedInputIndex
                                              - convolutionalConfig.padding;
                    if (inputIndex >= convolutionalConfig.inputLength) {
                        continue;
                    }

                    // This is the convolution version of W^T * delta: each
                    // output delta is spread back to the input cells that
                    // contributed to that output.
                    inputDelta.at({inputChannel, inputIndex})
                        += layerDelta.at({outputChannel, outputIndex})
                           * weights.at(
                               {outputChannel, inputChannel, kernelIndex});
                }
            }
        }
    }

    return inputDelta;
}

void ConvolutionalLayer::updateWeights(const Pattern &prev_activations,
                                       const Pattern &layerDelta,
                                       Scalar learningRate)
{
    requireInitialized();

    if (!prev_activations.hasShape(expectedInput)) {
        throw std::runtime_error("Previous activation shape does not match "
                                 "convolutional layer input shape.");
    }
    if (!layerDelta.hasShape(expectedOutput)) {
        throw std::runtime_error("Layer delta shape does not match "
                                 "convolutional layer output shape.");
    }

    Pattern weightGradients = Pattern::withShape(expectedWeightShape(),
                                                 Scalar{0});
    Pattern biasGradients = Pattern::withShape(expectedBiasShape(), Scalar{0});

    for (size_t outputChannel = 0;
         outputChannel < convolutionalConfig.outputChannels;
         ++outputChannel) {
        for (size_t outputIndex = 0; outputIndex < layerDelta.shape().at(1);
             ++outputIndex) {
            biasGradients.at(outputChannel) += layerDelta.at(
                {outputChannel, outputIndex});

            for (size_t inputChannel = 0;
                 inputChannel < convolutionalConfig.inputChannels;
                 ++inputChannel) {
                for (size_t kernelIndex = 0;
                     kernelIndex < convolutionalConfig.kernelSize;
                     ++kernelIndex) {
                    const size_t paddedInputIndex
                        = (outputIndex * convolutionalConfig.stride)
                          + kernelIndex;

                    if (paddedInputIndex < convolutionalConfig.padding) {
                        continue;
                    }

                    const size_t inputIndex = paddedInputIndex
                                              - convolutionalConfig.padding;
                    if (inputIndex >= convolutionalConfig.inputLength) {
                        continue;
                    }

                    weightGradients.at(
                        {outputChannel, inputChannel, kernelIndex})
                        += layerDelta.at({outputChannel, outputIndex})
                           * prev_activations.at({inputChannel, inputIndex});
                }
            }
        }
    }

    const auto updateValue = [this, learningRate](Scalar value,
                                                  Scalar gradient) {
        return trainableConfig.learningRule->updateWeight(value,
                                                          gradient,
                                                          learningRate);
    };

    weights = weights.zip(weightGradients, updateValue);

    if (hasBias()) {
        biases = biases.zipValues(biasGradients, updateValue);
    }
}
