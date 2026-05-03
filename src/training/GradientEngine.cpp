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
void fillFlattenBackwardPass(const Pattern &layerDelta,
                             Pattern &previousDelta)
{
    if (layerDelta.size() != previousDelta.size()) {
        throw std::runtime_error(
            "Layer delta size does not match previous activation size.");
    }

    for (size_t i = 0; i < layerDelta.size(); ++i) {
        previousDelta[i] = layerDelta[i];
    }
}

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
void fillParameterizedBackwardPass(const LayerType &layer,
                                   const Pattern &weights,
                                   const Pattern &layerDelta,
                                   const Pattern &layerInput,
                                   Pattern &output)
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

    if (!output.hasShape(layer.getExpectedInputShape())) {
        throw std::runtime_error("Backward output shape does not match layer input shape.");
    }

    const size_t rows = weights.shape().at(0);
    const size_t cols = weights.shape().at(1);
    for (size_t col = 0; col < cols; ++col) {
        Scalar sum = Scalar{};
        for (size_t row = 0; row < rows; ++row) {
            sum += weights.at({row, col}) * layerDelta.at(row);
        }
        output.at(col) = sum;
    }
}

void fillConvolutionalBackwardPass(const ConvolutionalLayer &layer,
                                   const Pattern &weights,
                                   const Pattern &layerDelta,
                                   const Pattern &layerInput,
                                   Pattern &inputDelta);

void fillBackwardThroughSkill(const Skill &skill,
                              const Pattern &layerDelta,
                              const Pattern &layerInput,
                              Pattern &previousDelta)
{
    const auto &layer = skill.layer();
    if (auto dense = layerAs<DenseLayer>(layer)) {
        fillParameterizedBackwardPass(dense->get(),
                                      skill.getParameters().weights,
                                      layerDelta,
                                      layerInput,
                                      previousDelta);
        return;
    }

    if (auto convolutional = layerAs<ConvolutionalLayer>(layer)) {
        fillConvolutionalBackwardPass(convolutional->get(),
                                      skill.getParameters().weights,
                                      layerDelta,
                                      layerInput,
                                      previousDelta);
        return;
    }

    if (auto hopfield = layerAs<HopfieldLayer>(layer)) {
        fillParameterizedBackwardPass(hopfield->get(),
                                      skill.getParameters().weights,
                                      layerDelta,
                                      layerInput,
                                      previousDelta);
        return;
    }

    if (layerAs<FlattenLayer>(layer)) {
        fillFlattenBackwardPass(layerDelta, previousDelta);
        return;
    }

    throw std::runtime_error(
        "Skill does not support feedforward backpropagation.");
}

void fillConvolutionalBackwardPass(const ConvolutionalLayer &layer,
                                   const Pattern &weights,
                                   const Pattern &layerDelta,
                                   const Pattern &layerInput,
                                   Pattern &inputDelta)
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

    if (!inputDelta.hasShape(layer.getExpectedInputShape())) {
        throw std::runtime_error("Backward output shape does not match convolutional input.");
    }

    for (size_t i = 0; i < inputDelta.size(); ++i) {
        inputDelta[i] = Scalar{};
    }

    const auto &recipe = layer.getConvolutionalRecipe();
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

}
} // namespace

void BackpropagationGradientEngine::computeLayerDeltas(
    const Model &network,
    const Batch &activations,
    const Batch &preActivations,
    const Pattern &outputError,
    Batch &layerDeltas) const
{
    if (layerDeltas.size() != network.numLayers()) {
        throw std::runtime_error("Layer delta buffer size does not match network layers.");
    }

    // Fill the caller-owned delta batch in place to avoid per-sample allocation.
    const Layer &lastLayer = network.getLayer(network.numLayers() - 1);
    if (!layerDeltas.back().hasShape(lastLayer.getOutputShape())) {
        throw std::runtime_error("Output layer delta buffer shape mismatch.");
    }
    if (!outputError.hasShape(lastLayer.getOutputShape())) {
        throw std::runtime_error("Output error shape does not match network output.");
    }

    const auto &outputActivation = lastLayer.getActivation();
    if (!outputActivation) {
        throw std::runtime_error("Activation function is not set for this layer.");
    }
    for (size_t i = 0; i < outputError.size(); ++i) {
        layerDeltas.back()[i]
            = outputError[i] * outputActivation->derivative(preActivations.back()[i]);
    }

    for (size_t layerIndex = network.numLayers() - 1; layerIndex > 0;
         --layerIndex) {
        fillBackwardThroughSkill(network.getSkill(layerIndex),
                                 layerDeltas.at(layerIndex),
                                 activations.at(layerIndex),
                                 layerDeltas.at(layerIndex - 1));

        const Layer &previousLayer = network.getLayer(layerIndex - 1);
        if (supportsParameterizedTraining(previousLayer)) {
            const auto &activation = previousLayer.getActivation();
            if (!activation) {
                throw std::runtime_error(
                    "Activation function is not set for this layer.");
            }
            for (size_t i = 0; i < layerDeltas.at(layerIndex - 1).size(); ++i) {
                layerDeltas.at(layerIndex - 1)[i]
                    *= activation->derivative(preActivations.at(layerIndex - 1)[i]);
            }
        }
    }
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
        Pattern previousDelta = Pattern::withShape(Shape(layerInput.shape()), Scalar{0});
        fillFlattenBackwardPass(layerDelta, previousDelta);
        return previousDelta;
    }

    throw std::runtime_error(
        "Layer does not support feedforward backpropagation.");
}

Pattern BackpropagationGradientEngine::backwardThroughSkill(
    const Skill &skill,
    const Pattern &layerDelta,
    const Pattern &layerInput) const
{
    Pattern previousDelta = Pattern::withShape(Shape(layerInput.shape()), Scalar{0});
    fillBackwardThroughSkill(skill, layerDelta, layerInput, previousDelta);
    return previousDelta;
}
