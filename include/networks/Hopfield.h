#pragma once
#include "base/Model.h"
#include "base/Layer.h"
#include "base/LearningRule.h"
#include "layers/HopfieldLayer.h"
#include "base/Types.h"

#include <cstddef> // for size_t
#include <memory>

class Hopfield : public Model
{
public:

    Hopfield() = default; // Default constructor creates a Hopfield network with no layers

    Hopfield(std::unique_ptr<Layer> newLayer)
        : Model(Layers{})
    {
        if (newLayer) {
            layers.push_back(std::move(newLayer));
        } else {
            throw std::runtime_error("Perceptron must have at least one layer.");
        }
    }

    ~Hopfield() override = default;

    void learn(const Patterns &inputs)
    {
        learn(inputs, {}, 1.0f, 10000); // Default learning rate and epochs
    }

    void learn(const Patterns &inputs,
        const Patterns &labels,
        Scalar learningRate = 0.1f,
        size_t epochs = 100000) override
    {
        (void)(epochs); // Unused parameter
        (void)(labels); // Unused parameter

        if (inputs.empty()) {
            throw std::runtime_error("Patterns vector is empty.");
        }

        for (auto &pattern : inputs) {
            for (auto &layer : layers) {
                layer->updateWeights(pattern, {}, learningRate);
            }
        }
    }

    virtual Pattern infer(const Pattern &input) override
    {
        if (layers.empty()) {
            throw std::runtime_error(
                "No layers exist in the model to perform inference.");
        }
        ;
        return layers.front()->infer(input);
    }

private:
};