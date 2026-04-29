#include "training/FeedforwardTrainer.h"

#include "base/Layer.h"
#include "base/Model.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/DenseLayer.h"
#include "layers/HopfieldLayer.h"
#include "training/GradientEngine.h"
#include "training/LayerParameterInitializer.h"
#include "training/Optimizer.h"

#include <functional>
#include <memory>
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

Pattern preActivationFor(const Layer &layer, const Pattern &input)
{
    if (auto dense = layerAs<const DenseLayer>(layer)) {
        dense->get().requireInitialized();
        if (input.empty()) {
            throw std::runtime_error("Input is empty");
        }

        Pattern sums = input.matVec(dense->get().getWeights());
        const auto &biases = dense->get().getBiases();
        return biases.empty() ? sums : sums + biases;
    }
    if (auto convolutional = layerAs<const ConvolutionalLayer>(layer)) {
        convolutional->get().requireInitialized();
        if (input.empty()) {
            throw std::runtime_error("Input is empty");
        }

        const auto &recipe = convolutional->get().getConvolutionalRecipe();
        Pattern result = input.conv1D(convolutional->get().getWeights(),
                                      recipe.stride,
                                      recipe.padding);
        const auto &biases = convolutional->get().getBiases();
        if (!biases.empty()) {
            for (size_t outputChannel = 0;
                 outputChannel < recipe.outputChannels;
                 ++outputChannel) {
                for (size_t outputIndex = 0;
                     outputIndex < result.shape().at(1);
                     ++outputIndex) {
                    result.at({outputChannel, outputIndex}) += biases.at(
                        outputChannel);
                }
            }
        }

        return result;
    }
    if (auto hopfield = layerAs<const HopfieldLayer>(layer)) {
        hopfield->get().requireInitialized();
        if (input.empty()) {
            throw std::runtime_error("Input is empty");
        }

        Pattern sums = input.matVec(hopfield->get().getWeights());
        const auto &biases = hopfield->get().getBiases();
        return biases.empty() ? sums : sums + biases;
    }

    throw std::runtime_error(
        "Feedforward training requires pre-activation support.");
}

Pattern activateFor(const Layer &layer, const Pattern &values)
{
    const auto &activation = layer.getActivation();
    if (!activation) {
        throw std::runtime_error(
            "Activation function is not set for this layer.");
    }

    return values.map(
        [&activation](Scalar value) { return (*activation)(value); });
}

bool supportsParameterizedTraining(const Layer &layer)
{
    return layerAs<const DenseLayer>(layer).has_value()
           || layerAs<const ConvolutionalLayer>(layer).has_value()
           || layerAs<const HopfieldLayer>(layer).has_value();
}

void validateTrainingData(const Model &network,
                          const Batch &inputs,
                          const Batch &labels)
{
    const Layers &layers = network.getLayers();
    if (layers.empty()) {
        throw std::runtime_error(
            "Cannot train feedforward network without layers.");
    }
    if (inputs.empty() || inputs.size() != labels.size()) {
        throw std::runtime_error(
            "Inputs and labels must be non-empty and have the same size.");
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs.at(i).hasShape(layers.front()->getInputShape())) {
            throw std::runtime_error(
                "Training input shape does not match network input shape.");
        }
        if (!labels.at(i).hasShape(layers.back()->getOutputShape())) {
            throw std::runtime_error(
                "Training label shape does not match network output shape.");
        }
    }
}

void initializeTrainingBuffers(const Model &network,
                               Batch &activations,
                               Batch &preActivations)
{
    const Layers &layers = network.getLayers();
    activations = Batch(network.numLayers() + 1);
    preActivations = Batch(network.numLayers());
    activations.at(0) = Pattern::withShape(layers.front()->getInputShape(),
                                           Scalar{0});

    for (size_t layerIndex = 0; layerIndex < network.numLayers(); ++layerIndex) {
        activations.at(layerIndex + 1)
            = Pattern::withShape(layers.at(layerIndex)->getOutputShape(),
                                 Scalar{0});
        preActivations.at(layerIndex)
            = Pattern::withShape(layers.at(layerIndex)->getOutputShape(),
                                 Scalar{0});
    }
}

void forward(Model &network,
             const Pattern &input,
             Batch &activations,
             Batch &preActivations)
{
    Pattern current = input;
    activations.at(0) = current;

    for (size_t layerIndex = 0; layerIndex < network.numLayers(); ++layerIndex) {
        const auto &layer = network.getLayer(layerIndex);
        if (supportsParameterizedTraining(layer)) {
            preActivations.at(layerIndex) = preActivationFor(layer, current);
            current = activateFor(layer, preActivations.at(layerIndex));
        } else {
            current = layer.infer(current);
            preActivations.at(layerIndex) = current;
        }
        activations.at(layerIndex + 1) = current;
    }
}

Pattern lossDerivative(const Pattern &output, const Pattern &target)
{
    return output - target;
}

void backpropagation(Model &network,
                     const Batch &activations,
                     const Batch &preActivations,
                     const Pattern &outputError,
                     const GradientEngine &gradientEngine,
                     const Optimizer &optimizer,
                     Scalar learningRate)
{
    const Batch layerDeltas = gradientEngine.computeLayerDeltas(network,
                                                                activations,
                                                                preActivations,
                                                                outputError);

    optimizer.step(network, activations, layerDeltas, learningRate);
}
} // namespace

void FeedforwardTrainer::learn(Model &network,
                               const Batch &inputs,
                               const Batch &labels,
                               Scalar learningRate,
                               size_t epochs)
{
    validateTrainingData(network, inputs, labels);

    Batch activations;
    Batch preActivations;
    initializeModelParameters(network);
    initializeTrainingBuffers(network, activations, preActivations);
    const LearningRuleOptimizer optimizer{
        std::make_shared<SGDRule<Scalar>>()};
    const BackpropagationGradientEngine gradientEngine;

    std::cout << "Training feedforward Network..." << std::endl;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t sampleIndex = 0; sampleIndex < inputs.size();
             ++sampleIndex) {
            forward(network,
                    inputs.at(sampleIndex),
                    activations,
                    preActivations);
            const Pattern outputError = lossDerivative(activations.back(),
                                                       labels.at(sampleIndex));
            backpropagation(network,
                            activations,
                            preActivations,
                            outputError,
                            gradientEngine,
                            optimizer,
                            learningRate);
        }
    }
}
