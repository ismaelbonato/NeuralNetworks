#pragma once

#include "base/Types.h"

class Layer;
class Model;

class GradientEngine
{
public:
    GradientEngine() = default;
    virtual ~GradientEngine() = default;

    virtual Batch computeLayerDeltas(const Model &network,
                                     const Batch &activations,
                                     const Batch &preActivations,
                                     const Pattern &outputError) const = 0;
};

class BackpropagationGradientEngine : public GradientEngine
{
public:
    Batch computeLayerDeltas(const Model &network,
                             const Batch &activations,
                             const Batch &preActivations,
                             const Pattern &outputError) const override;

    Pattern backwardThroughLayer(const Layer &layer,
                                 const Pattern &layerDelta,
                                 const Pattern &layerInput) const;

    Pattern activationDerivatives(const Layer &layer,
                                  const Pattern &values) const;

private:
    Pattern applyActivationDerivative(const Layer &layer,
                                      const Pattern &outputGradient,
                                      const Pattern &preActivation) const;
};
