#include "training/FeedforwardTrainer.h"

#include "base/Layer.h"
#include "base/Model.h"

#include <iostream>
#include <stdexcept>

namespace
{
void validateTrainingData(const Model &network,
                          const Batch &inputs,
                          const Batch &labels)
{
    const Layers &layers = network.getLayers();
    if (layers.empty()) {
        throw std::runtime_error("Cannot train feedforward network without layers.");
    }
    if (inputs.empty() || inputs.size() != labels.size()) {
        throw std::runtime_error("Inputs and labels must be non-empty and have the same size.");
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].size() != layers.front()->getInputSize()) {
            throw std::runtime_error("Training input size does not match network input size.");
        }
        if (labels[i].size() != layers.back()->getOutputSize()) {
            throw std::runtime_error("Training label size does not match network output size.");
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
    activations[0] = Pattern::vector(layers.front()->getInputSize(), Scalar{0});

    for (size_t layerIndex = 0; layerIndex < network.numLayers(); ++layerIndex) {
        activations[layerIndex + 1] =
            Pattern::vector(layers[layerIndex]->getOutputSize(), Scalar{0});
        preActivations[layerIndex] =
            Pattern::vector(layers[layerIndex]->getOutputSize(), Scalar{0});
    }
}

void forward(Model &network,
             const Pattern &input,
             Batch &activations,
             Batch &preActivations)
{
    Pattern current = input;
    activations[0] = current;

    for (size_t layerIndex = 0; layerIndex < network.numLayers(); ++layerIndex) {
        auto &layer = network.getLayer(layerIndex);
        preActivations[layerIndex] = layer->weightedSum(current);
        current = layer->activate(preActivations[layerIndex]);
        activations[layerIndex + 1] = current;
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
                     Scalar learningRate)
{
    Batch layerDeltas(network.numLayers());

    layerDeltas.back() =
        outputError * network.getLayer(network.numLayers() - 1)->activationDerivatives(
                          preActivations.back());

    for (size_t layerIndex = network.numLayers() - 1; layerIndex > 0; --layerIndex) {
        layerDeltas[layerIndex - 1] =
            network.getLayer(layerIndex)->backwardPass(layerDeltas[layerIndex],
                                                       preActivations[layerIndex - 1]);
    }

    for (size_t layerIndex = 0; layerIndex < network.numLayers(); ++layerIndex) {
        network.getLayer(layerIndex)->updateWeights(activations[layerIndex],
                                                    layerDeltas[layerIndex],
                                                    learningRate);
    }
}
}

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
        for (size_t sampleIndex = 0; sampleIndex < inputs.size(); ++sampleIndex) {
            forward(network, inputs[sampleIndex], activations, preActivations);
            const Pattern outputError =
                lossDerivative(activations.back(), labels[sampleIndex]);
            backpropagation(network, activations, preActivations, outputError, learningRate);
        }
    }
}
