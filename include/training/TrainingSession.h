#pragma once

#include "base/Tensor.h"
#include "base/Types.h"

class Model;

class TrainingSession
{
public:
    explicit TrainingSession(Model &newNetwork);

    Model &model();
    const Model &model() const;

    void initializeForwardBuffers();
    void setLayerDeltas(Batch newLayerDeltas);

    Batch &activations();
    const Batch &activations() const;
    Batch &preActivations();
    const Batch &preActivations() const;
    Batch &layerDeltas();
    const Batch &layerDeltas() const;

private:
    Model &network;
    Batch activationBuffer;
    Batch preActivationBuffer;
    Batch layerDeltaBuffer;
};
