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

    Batch &activations();
    const Batch &activations() const;
    Batch &preActivations();
    const Batch &preActivations() const;
    Batch &layerDeltas();
    const Batch &layerDeltas() const;
    Pattern &outputError();
    const Pattern &outputError() const;

private:
    Model &network;
    Batch activationBuffer;
    Batch preActivationBuffer;
    Batch layerDeltaBuffer;
    Pattern outputErrorBuffer;
};
