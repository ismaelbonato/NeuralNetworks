#pragma once
#include "base/Layer.h"
#include "base/Model.h"
#include "base/GradientBaseModel.h"
#include "networks/DenseLayer.h"
#include "base/Types.h"

class FeedforwardNetwork : public GradientBaseModel
{
public:

    // Default constructor creates a network with a single layer
    FeedforwardNetwork() = default; // No layers by default, can be extended later

    FeedforwardNetwork(Layers newlayers)
    : GradientBaseModel(std::move(newlayers))
    {}

    ~FeedforwardNetwork() override = default;

    void addLayer(std::unique_ptr<Layer> layer) override 
    {
        // Update the activate and preActivations vectors
        if (activate.empty()) {
            activate.emplace_back(Pattern(layer->getOutputSize(), 0.0f));
        } 
        activate.emplace_back(Pattern(layer->getOutputSize()));
        preActivations.emplace_back(Pattern(layer->getInputSize(), 0.0f));

        GradientBaseModel::addLayer(std::move(layer));
    }

    // Call the learning rule's learn method supervised
    void learn(const Patterns &inputs,
               const Patterns &labels,
               Scalar learningRate = 0.1f,
               size_t epochs = 100000) override
    {
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