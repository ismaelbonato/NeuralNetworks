#pragma once

#include "base/Types.h"

class Layer;
class Model;
class Skill;

class GradientEngine
{
public:
    GradientEngine() = default;
    virtual ~GradientEngine() = default;

    virtual void computeLayerDeltas(const Model &network,
                                    const Batch &activations,
                                    const Batch &preActivations,
                                    const Pattern &outputError,
                                    Batch &layerDeltas) const = 0;
};

class BackpropagationGradientEngine : public GradientEngine
{
public:
    void computeLayerDeltas(const Model &network,
                            const Batch &activations,
                            const Batch &preActivations,
                            const Pattern &outputError,
                            Batch &layerDeltas) const override;

    Pattern backwardThroughLayer(const Layer &layer,
                                 const Pattern &layerDelta,
                                 const Pattern &layerInput) const;
    Pattern backwardThroughSkill(const Skill &skill,
                                 const Pattern &layerDelta,
                                 const Pattern &layerInput) const;
};
