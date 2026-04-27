#include "training/FeedforwardTrainer.h"

#include "base/Layer.h"
#include "base/Model.h"
#include "layers/FlattenLayer.h"

#include <iostream>
#include <stdexcept>

namespace {
TrainableLayer &requireTrainable(Layer &layer)
{
    auto *trainable = dynamic_cast<TrainableLayer *>(&layer);
    if (trainable == nullptr) {
        throw std::runtime_error(
            "Feedforward training requires trainable layer operations.");
    }

    return *trainable;
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
        const auto *trainable = dynamic_cast<const TrainableLayer *>(&layer);
        if (trainable != nullptr) {
            preActivations.at(layerIndex) = trainable->preActivation(current);
            current = trainable->activate(preActivations.at(layerIndex));
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

Pattern backwardThroughLayer(const Layer &layer,
                             const Pattern &layerDelta,
                             const Pattern &layerInput)
{
    const auto *trainable = dynamic_cast<const TrainableLayer *>(&layer);
    if (trainable != nullptr) {
        return trainable->backwardPass(layerDelta, layerInput);
    }

    const auto *flatten = dynamic_cast<const FlattenLayer *>(&layer);
    if (flatten != nullptr) {
        return flatten->backwardPass(layerDelta, layerInput);
    }

    throw std::runtime_error(
        "Layer does not support feedforward backpropagation.");
}

Pattern applyActivationDerivative(const Layer &layer,
                                  const Pattern &outputGradient,
                                  const Pattern &preActivation)
{
    const auto *trainable = dynamic_cast<const TrainableLayer *>(&layer);
    if (trainable == nullptr) {
        return outputGradient;
    }

    return outputGradient * trainable->activationDerivatives(preActivation);
}

void updateTrainableLayers(Model &network,
                           const Batch &activations,
                           const Batch &layerDeltas,
                           Scalar learningRate)
{
    for (size_t layerIndex = 0; layerIndex < network.numLayers(); ++layerIndex) {
        auto *trainable = dynamic_cast<TrainableLayer *>(
            &network.getLayer(layerIndex));
        if (trainable != nullptr) {
            trainable->updateWeights(activations.at(layerIndex),
                                     layerDeltas.at(layerIndex),
                                     learningRate);
        }
    }
}

void backpropagation(Model &network,
                     const Batch &activations,
                     const Batch &preActivations,
                     const Pattern &outputError,
                     Scalar learningRate)
{
    Batch layerDeltas(network.numLayers());

    const auto &outputLayer = requireTrainable(
        network.getLayer(network.numLayers() - 1));
    layerDeltas.back() = outputError
                         * outputLayer.activationDerivatives(
                             preActivations.back());

    for (size_t layerIndex = network.numLayers() - 1; layerIndex > 0;
         --layerIndex) {
        const Pattern previousLayerOutputGradient
            = backwardThroughLayer(network.getLayer(layerIndex),
                                   layerDeltas.at(layerIndex),
                                   activations.at(layerIndex));

        layerDeltas.at(layerIndex - 1)
            = applyActivationDerivative(network.getLayer(layerIndex - 1),
                                        previousLayerOutputGradient,
                                        preActivations.at(layerIndex - 1));
    }

    updateTrainableLayers(network, activations, layerDeltas, learningRate);
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
    initializeTrainingBuffers(network, activations, preActivations);

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
                            learningRate);
        }
    }
}
