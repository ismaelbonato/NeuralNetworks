#pragma once
#include "base/Model.h"
#include "base/Layer.h"
#include "base/LearningRule.h"
#include "networks/HopfieldLayer.h"
#include "base/Types.h"

#include <cstddef> // for size_t
#include <memory>

class Hopfield : public Model
{
public:
    Hopfield(size_t n)
    {
        layers.emplace_back(std::make_unique<HopfieldLayer>(
            std::make_shared<HebbianRule<Scalar>>(),
            std::make_shared<StepPolarActivation<Scalar>>(),
            n,
            n));
    }

    // Constructor with custom learning rule
    Hopfield(const std::shared_ptr<LearningRule<Scalar>> &newRule,
             const std::shared_ptr<ActivationFunction<Scalar>> &activationFunction,
             size_t n)
    {
        layers.emplace_back(
            std::make_unique<HopfieldLayer>(newRule, activationFunction, n, n));
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