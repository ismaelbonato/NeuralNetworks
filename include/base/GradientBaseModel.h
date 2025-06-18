#pragma once

#include "LayeredModel.h"


class GradientBaseModel : public LayeredModel
{
protected:
    Patterns activate;       // Store activations for each layer
    Patterns preActivations; // Store pre-activations for each layer

public:
    GradientBaseModel() = default;

    GradientBaseModel(Layers newlayers)
        : LayeredModel(std::move(newlayers))
        , activate(numLayers() + 1) // +1 for the input layer
        , preActivations(numLayers()) // No pre-activation for the input layer
    {}

    virtual ~GradientBaseModel() = default;

    // Add a layer to the model
    void addLayer(std::unique_ptr<Layer> layer) override
    {
        LayeredModel::addLayer(std::move(layer));
    }

    // Remove a layer by index
    void removeLayer(size_t index) override
    {
        LayeredModel::removeLayer(index);
    }

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




