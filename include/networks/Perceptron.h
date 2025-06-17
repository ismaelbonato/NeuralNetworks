#pragma once

#include "base/LayeredModel.h"
#include "networks/DenseLayer.h"
#include "base/Types.h"

#include <memory>

class Perceptron : public LayeredModel
{
public:
    Perceptron() = default; // No layers by default, can be extended later
    
    Perceptron(std::unique_ptr<Layer> newLayer)
        : LayeredModel(Layers{})
    {
        if (newLayer) {
            layers.push_back(std::move(newLayer));
        } else {
            throw std::runtime_error("Perceptron must have at least one layer.");
        }
    }

    ~Perceptron() override = default;

    void learn(const Patterns &inputs,
               const Patterns &labels,
               Scalar learningRate = 0.1f,
               size_t epochs = 10000) override
    {
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
        return {target.front() - activated.front()};
    }
};