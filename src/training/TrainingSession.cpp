#include "training/TrainingSession.h"

#include "base/Model.h"

#include <utility>

TrainingSession::TrainingSession(Model &newNetwork)
    : network(newNetwork)
{}

Model &TrainingSession::model()
{
    return network;
}

const Model &TrainingSession::model() const
{
    return network;
}

void TrainingSession::initializeForwardBuffers()
{
    activationBuffer = Batch(network.numLayers() + 1);
    preActivationBuffer = Batch(network.numLayers());
    layerDeltaBuffer = Batch(network.numLayers());

    activationBuffer.at(0) = Pattern::withShape(
        network.getLayer(0).getInputShape(),
        Scalar{0});

    for (size_t layerIndex = 0; layerIndex < network.numLayers();
         ++layerIndex) {
        const auto &layer = network.getLayer(layerIndex);
        activationBuffer.at(layerIndex + 1)
            = Pattern::withShape(layer.getOutputShape(), Scalar{0});
        preActivationBuffer.at(layerIndex)
            = Pattern::withShape(layer.getOutputShape(), Scalar{0});
        layerDeltaBuffer.at(layerIndex)
            = Pattern::withShape(layer.getOutputShape(), Scalar{0});
    }
    outputErrorBuffer = Pattern::withShape(
        network.getLayer(network.numLayers() - 1).getOutputShape(),
        Scalar{0});
}

Batch &TrainingSession::activations()
{
    return activationBuffer;
}

const Batch &TrainingSession::activations() const
{
    return activationBuffer;
}

Batch &TrainingSession::preActivations()
{
    return preActivationBuffer;
}

const Batch &TrainingSession::preActivations() const
{
    return preActivationBuffer;
}

Batch &TrainingSession::layerDeltas()
{
    return layerDeltaBuffer;
}

const Batch &TrainingSession::layerDeltas() const
{
    return layerDeltaBuffer;
}

Pattern &TrainingSession::outputError()
{
    return outputErrorBuffer;
}

const Pattern &TrainingSession::outputError() const
{
    return outputErrorBuffer;
}
