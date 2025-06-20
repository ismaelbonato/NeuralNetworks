#pragma once
#include "base/Layer.h"
#include "base/Model.h"
#include "base/GradientBaseModel.h"
#include "layers/DenseLayer.h"
#include "base/Types.h"

#include <initializer_list>

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
        // Update the activate and preActivations vectors
        if (activate.empty()) {
            activate.emplace_back(Pattern(layers.front()->getInputSize(), 0.0f));
        }
        
        for (auto &&l : layers)
        {
            activate.emplace_back(Pattern(l->getOutputSize(), 0.0f));
            preActivations.emplace_back(Pattern(l->getInputSize(), 0.0f));
        }

        std::cout << "Training feedforward Network..." << std::endl;
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                forward(inputs[i]);
                Pattern delta = computeError(labels[i]);
                backpropagation(delta, learningRate);
            }
        }
    }
};