#include "networks/FeedForward.h"

#include <iostream>
#include <stdexcept>

Feedforward::Feedforward() = default;

Feedforward::Feedforward(Layers &newlayers)
    : GradientBaseModel(newlayers)
{}

Feedforward::Feedforward(const std::initializer_list<std::shared_ptr<Layer>> &newlayers)
    : GradientBaseModel(newlayers)
{}

Feedforward::~Feedforward() = default;

void Feedforward::learn(const Patterns &inputs,
                        const Patterns &labels,
                        Scalar learningRate,
                        size_t epochs)
{
    if (layers.empty()) {
        throw std::runtime_error("Cannot train feedforward network without layers.");
    }
    if (inputs.empty() || inputs.size() != labels.size()) {
        throw std::runtime_error("Inputs and labels must be non-empty and have the same size.");
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].size() != layers.front()->getInputSize()) {
            throw std::runtime_error("Training input size does not match network input size.");
        }
        if (labels[i].size() != layers.back()->getOutputSize()) {
            throw std::runtime_error("Training label size does not match network output size.");
        }
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
