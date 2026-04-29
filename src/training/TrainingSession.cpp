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
    weightedInputBuffer = Batch(network.numLayers());
    layerDeltaBuffer = Batch(network.numLayers());

    activationBuffer.at(0) = Pattern::withShape(
        network.getLayer(0).getInputShape(),
        Scalar{0});

    for (size_t layerIndex = 0; layerIndex < network.numLayers();
         ++layerIndex) {
        const auto &layer = network.getLayer(layerIndex);
        activationBuffer.at(layerIndex + 1)
            = Pattern::withShape(layer.getOutputShape(), Scalar{0});
        weightedInputBuffer.at(layerIndex)
            = Pattern::withShape(layer.getOutputShape(), Scalar{0});
    }
}

void TrainingSession::setLayerDeltas(Batch newLayerDeltas)
{
    layerDeltaBuffer = std::move(newLayerDeltas);
}

Batch &TrainingSession::activations()
{
    return activationBuffer;
}

const Batch &TrainingSession::activations() const
{
    return activationBuffer;
}

Batch &TrainingSession::weightedInputs()
{
    return weightedInputBuffer;
}

const Batch &TrainingSession::weightedInputs() const
{
    return weightedInputBuffer;
}

Batch &TrainingSession::layerDeltas()
{
    return layerDeltaBuffer;
}

const Batch &TrainingSession::layerDeltas() const
{
    return layerDeltaBuffer;
}
