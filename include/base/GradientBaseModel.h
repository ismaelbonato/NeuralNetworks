#pragma once

#include "LayeredModel.h"
#include <stdexcept>
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
        if (layers.empty()) {
            throw std::runtime_error("Cannot backpropagate without layers.");
        }

        Patterns layerDeltas(numLayers());

        // delta^L = dC/da^L * sigma'(z^L)
        layerDeltas.back() =
            outputError * layers.back()->activationDerivatives(preActivations.back());

        // delta^l = ((W^(l+1))^T * delta^(l+1)) * sigma'(z^l)
        for (size_t l = numLayers() - 1; l > 0; --l) {
            layerDeltas[l - 1] =
                getLayer(l)->backwardPass(layerDeltas[l], preActivations[l - 1]);
        }

        // dC/dW^l = delta^l * (a^(l-1))^T, dC/db^l = delta^l
        for (size_t l = 0; l < numLayers(); ++l) {
            getLayer(l)->updateWeights(activate[l], layerDeltas[l], rate);
        }
    }

    virtual inline Pattern lossDerivative(const Pattern &output,
                                          const Pattern &target)
    {
        return output - target;
    }
};
