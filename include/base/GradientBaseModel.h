#pragma once

#include "LayeredModel.h"
#include <vector>

class GradientBaseModel : public LayeredModel
{
protected:
    Patterns activate;       // Store activations for each layer
    Patterns preActivations; // Store pre-activations for each layer

public:
    GradientBaseModel() = default;

    GradientBaseModel(Layers &newlayers)
        : LayeredModel(newlayers)
    {}

    GradientBaseModel(const std::initializer_list<std::shared_ptr<Layer>> &newlayers)
        : LayeredModel(newlayers)
    {}

    virtual ~GradientBaseModel() = default;

    virtual void forward(const Pattern &input)
    {
        // Forward pass: store activations and pre-activations
        Pattern current = input;
        activate[0] = current;

        for (size_t i = 0; i < numLayers(); i++) {
            preActivations[i] = getLayer(i)->weightedSum(current);
            current = getLayer(i)->activate(preActivations[i]);
            activate[i + 1] = current;
        }
    }

    virtual inline Pattern computeOutputError(const Pattern &target)
    {
        return lossDerivative(activate.back(), target);
    }

    virtual void backpropagation(const Pattern &outputError, const Scalar rate)
    {
        Pattern currentLayerDelta =
            outputError * layers.back()->activationDerivatives(preActivations.back());

        // Backward pass: propagate error and update weights
        for (size_t l = numLayers(); l-- > 0;) {
            Pattern previousLayerDelta;

            if (l > 0) {
                previousLayerDelta =
                    getLayer(l)->backwardPass(currentLayerDelta,
                                              preActivations[l - 1]);
            }

            getLayer(l)->updateWeights(activate[l], currentLayerDelta, rate);

            if (l > 0) {
                currentLayerDelta = previousLayerDelta;
            }
        }
    }

    virtual inline Pattern lossDerivative(const Pattern &output,
                                          const Pattern &target)
    {
        return output - target;
    }
};
