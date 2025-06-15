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
        : Model(n)
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
        : Model(n)
    {
        layers.emplace_back(
            std::make_unique<HopfieldLayer>(newRule, activationFunction, n, n));
    }

    ~Hopfield() override = default;

    void learn(const Patterns &patterns) override
    {
        if (patterns.empty()) {
            throw std::runtime_error("Patterns vector is empty.");
        }

        for (auto &pattern : patterns) {
            for (auto &layer : layers) {
                layer->updateWeights(pattern, {}, 1.0f);
            }
        }
    }

    void learn(const Patterns &,
               const Patterns &,
               Scalar,
               size_t) override
    {
        std::cerr << "Learn method not implemented for Hopfield network."
                  << std::endl;
        throw std::runtime_error(
            "Learn method not implemented for Hopfield network.");
    }

private:
};