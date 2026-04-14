#pragma once

#include "base/LayeredModel.h"
#include "layers/DenseLayer.h"
#include "base/Types.h"

#include <memory>
#include <stdexcept>

class Perceptron : public LayeredModel
{
public:
    Perceptron() = default; // No layers by default, can be extended later
    
    Perceptron(const std::shared_ptr<Layer> &newLayer)
        : LayeredModel({newLayer})
    {}

    ~Perceptron() override = default;

    void learn(const Patterns &inputs,
               const Patterns &labels,
               Scalar learningRate = 0.1f,
               size_t epochs = 10000) override
    {
        if (layers.empty()) {
            throw std::runtime_error("Cannot train perceptron without a layer.");
        }
        if (layers.front()->getOutputSize() != 1) {
            throw std::runtime_error("Perceptron supports exactly one output.");
        }
        if (inputs.empty() || inputs.size() != labels.size()) {
            throw std::runtime_error("Inputs and labels must be non-empty and have the same size.");
        }

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                Pattern activated = layers.front()->infer(inputs[i]);
                Pattern error = computeError(labels[i], activated);
                layers.front()->updateWeights(inputs[i], error, learningRate);
            }
        }
    }

    Pattern computeError(const Pattern &target, const Pattern &activated) const
    {
        return Pattern{target.front() - activated.front()};
    }
};