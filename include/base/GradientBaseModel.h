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

    GradientBaseModel(Layers newlayers)
        : LayeredModel(std::move(newlayers))
    {}

    template<typename... Args>
    GradientBaseModel(Args&... ls)
        : LayeredModel(ls...)
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

    virtual inline Pattern computeError(const Pattern &target)
    {
        return lossDerivative(activate.back(), target) * layers.back()->activationDerivatives(
                                   preActivations.back());
    }

    virtual void backpropagation(Pattern &delta, const Scalar rate)
    {
        // Backward pass: update weights and propagate error
        for (size_t l = numLayers(); l-- > 0;) {
            getLayer(l)->updateWeights(activate[l], delta, rate);
            if (l > 0) {
                delta = getLayer(l)->backwardPass(delta, preActivations[l - 1]);
            }
        }
    }

    virtual inline Pattern lossDerivative(const Pattern &output, const Pattern &target)
    {
        return output - target;
    }

};




