#include "training/GradientEngine.h"

#include "base/Layer.h"
#include "base/Model.h"
#include "base/Skill.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/DenseLayer.h"
#include "layers/FlattenLayer.h"
#include "layers/HopfieldLayer.h"

#include <functional>
#include <optional>
#include <stdexcept>
#include <typeinfo>

namespace {
template<typename LayerType>
std::optional<std::reference_wrapper<const LayerType>> layerAs(
    const Layer &layer)
{
    try {
        return std::cref(dynamic_cast<const LayerType &>(layer));
    } catch (const std::bad_cast &) {
        return std::nullopt;
    }
}

bool supportsParameterizedTraining(const Layer &layer)
{
    return layerAs<DenseLayer>(layer).has_value()
           || layerAs<ConvolutionalLayer>(layer).has_value()
           || layerAs<HopfieldLayer>(layer).has_value();
}

template<typename LayerType>
Pattern parameterizedBackwardPass(const LayerType &layer,
                                  const Pattern &weights,
                                  const Pattern &layerDelta,
                                  const Pattern &layerInput)
{
    if (!weights.hasShape(
            static_cast<const Layer &>(layer).expectedWeightShape())) {
        throw std::runtime_error(
            "Layer weights shape does not match layer recipe.");
    }
    if (!layerDelta.hasShape(layer.getExpectedOutputShape())) {
        throw std::runtime_error(
            "Layer delta shape does not match layer output shape.");
    }
    if (!layerInput.hasShape(layer.getExpectedInputShape())) {
        throw std::runtime_error(
            "Layer input shape does not match layer input shape.");
    }

    return weights.transposedMatVec(layerDelta);
}

Pattern convolutionalBackwardPass(const ConvolutionalLayer &layer,
                                  const Pattern &weights,
                                  const Pattern &layerDelta,
                                  const Pattern &layerInput)
{
    if (!weights.hasShape(
            static_cast<const Layer &>(layer).expectedWeightShape())) {
        throw std::runtime_error(
            "Layer weights shape does not match layer recipe.");
    }

    if (!layerDelta.hasShape(layer.getExpectedOutputShape())) {
        throw std::runtime_error("Layer delta shape does not match "
                                 "convolutional layer output shape.");
    }
    if (!layerInput.hasShape(layer.getExpectedInputShape())) {
        throw std::runtime_error("Layer input shape does not match "
                                 "convolutional layer input shape.");
    }

    const auto &recipe = layer.getConvolutionalRecipe();
    Pattern inputDelta = Pattern::withShape(layer.getExpectedInputShape(),
                                            Scalar{0});

    for (size_t outputChannel = 0; outputChannel < recipe.outputChannels;
         ++outputChannel) {
        for (size_t outputIndex = 0; outputIndex < layerDelta.shape().at(1);
             ++outputIndex) {
            for (size_t inputChannel = 0; inputChannel < recipe.inputChannels;
                 ++inputChannel) {
                for (size_t kernelIndex = 0; kernelIndex < recipe.kernelSize;
                     ++kernelIndex) {
                    const size_t paddedInputIndex
                        = (outputIndex * recipe.stride) + kernelIndex;

                    if (paddedInputIndex < recipe.padding) {
                        continue;
                    }

                    const size_t inputIndex = paddedInputIndex
                                              - recipe.padding;
                    if (inputIndex >= recipe.inputLength) {
                        continue;
                    }

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

Pattern flattenBackwardPass(const Pattern &layerDelta,
                            const Pattern &preActivation)
{
    if (layerDelta.size() != preActivation.size()) {
        throw std::runtime_error(
            "Layer delta size does not match previous activation size.");
    }

    Pattern previousDelta = layerDelta;
    previousDelta.reshape(Shape(preActivation.shape()));
    return previousDelta;
}
} // namespace

Batch BackpropagationGradientEngine::computeLayerDeltas(
    const Model &network,
    const Batch &activations,
    const Batch &preActivations,
    const Pattern &outputError) const
{
    Batch layerDeltas(network.numLayers());

    layerDeltas.back() = outputError
                         * activationDerivatives(
                             network.getLayer(network.numLayers() - 1),
                             preActivations.back());

    for (size_t layerIndex = network.numLayers() - 1; layerIndex > 0;
         --layerIndex) {
        const Pattern previousLayerOutputGradient
            = backwardThroughSkill(network.getSkill(layerIndex),
                                   layerDeltas.at(layerIndex),
                                   activations.at(layerIndex));

        layerDeltas.at(layerIndex - 1)
            = applyActivationDerivative(network.getLayer(layerIndex - 1),
                                        previousLayerOutputGradient,
                                        preActivations.at(layerIndex - 1));
    }

    return layerDeltas;
}

Pattern BackpropagationGradientEngine::backwardThroughLayer(
    const Layer &layer,
    const Pattern &layerDelta,
    const Pattern &layerInput) const
{
    if (auto dense = layerAs<DenseLayer>(layer)) {
        (void)dense;
        throw std::runtime_error(
            "Layer backpropagation requires skill parameters.");
    }

    if (auto convolutional = layerAs<ConvolutionalLayer>(layer)) {
        (void)convolutional;
        throw std::runtime_error(
            "Layer backpropagation requires skill parameters.");
    }

    if (auto hopfield = layerAs<HopfieldLayer>(layer)) {
        (void)hopfield;
        throw std::runtime_error(
            "Layer backpropagation requires skill parameters.");
    }

    if (layerAs<FlattenLayer>(layer)) {
        return flattenBackwardPass(layerDelta, layerInput);
    }

    throw std::runtime_error(
        "Layer does not support feedforward backpropagation.");
}

Pattern BackpropagationGradientEngine::backwardThroughSkill(
    const Skill &skill,
    const Pattern &layerDelta,
    const Pattern &layerInput) const
{
    const auto &layer = skill.layer();
    if (auto dense = layerAs<DenseLayer>(layer)) {
        return parameterizedBackwardPass(dense->get(),
                                         skill.getParameters().weights,
                                         layerDelta,
                                         layerInput);
    }

    if (auto convolutional = layerAs<ConvolutionalLayer>(layer)) {
        return convolutionalBackwardPass(convolutional->get(),
                                         skill.getParameters().weights,
                                         layerDelta,
                                         layerInput);
    }

    if (auto hopfield = layerAs<HopfieldLayer>(layer)) {
        return parameterizedBackwardPass(hopfield->get(),
                                         skill.getParameters().weights,
                                         layerDelta,
                                         layerInput);
    }

    if (layerAs<FlattenLayer>(layer)) {
        return flattenBackwardPass(layerDelta, layerInput);
    }

    throw std::runtime_error(
        "Skill does not support feedforward backpropagation.");
}

Pattern BackpropagationGradientEngine::activationDerivatives(
    const Layer &layer,
    const Pattern &values) const
{
    const auto &activation = layer.getActivation();
    if (!activation) {
        throw std::runtime_error(
            "Activation function is not set for this layer.");
    }

    return values.map([&activation](Scalar value) {
        return activation->derivative(value);
    });
}

Pattern BackpropagationGradientEngine::applyActivationDerivative(
    const Layer &layer,
    const Pattern &outputGradient,
    const Pattern &preActivation) const
{
    if (!supportsParameterizedTraining(layer)) {
        return outputGradient;
    }

    return outputGradient * activationDerivatives(layer, preActivation);
}
