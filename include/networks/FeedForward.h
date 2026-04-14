#pragma once
#include "base/Layer.h"
#include "base/Model.h"
#include "base/GradientBaseModel.h"
#include "layers/DenseLayer.h"
#include "base/Types.h"

#include <initializer_list>
#include <stdexcept>

class Feedforward : public GradientBaseModel
{
public:

    // Default constructor creates a network with a single layer
    Feedforward() = default; // No layers by default, can be extended later


    Feedforward(Layers &newlayers)
        : GradientBaseModel(newlayers)
    {}

    Feedforward(const std::initializer_list<std::shared_ptr<Layer>> &newlayers)
        : GradientBaseModel(newlayers)
    {}

    ~Feedforward() override = default;

    // Call the learning rule's learn method supervised
    void learn(const Patterns &inputs,
               const Patterns &labels,
               Scalar learningRate = 0.1f,
               size_t epochs = 100000) override
    {
        if (layers.empty()) {
            throw std::runtime_error("Cannot train feedforward network without layers.");
        }
        if (inputs.empty() || inputs.size() != labels.size()) {
            throw std::runtime_error("Inputs and labels must be non-empty and have the same size.");
        }

        activate = Patterns(numLayers() + 1);
        preActivations = Patterns(numLayers());
        activate[0] = Pattern(layers.front()->getInputSize(), 0.0f);

        for (size_t l = 0; l < numLayers(); ++l) {
            activate[l + 1] = Pattern(layers[l]->getOutputSize(), 0.0f);
            preActivations[l] = Pattern(layers[l]->getOutputSize(), 0.0f);
        }

        std::cout << "Training feedforward Network..." << std::endl;
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                forward(inputs[i]);
                Pattern outputError = computeOutputError(labels[i]);
                backpropagation(outputError, learningRate);
            }
        }
    }
};