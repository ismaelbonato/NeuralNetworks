#pragma once
#include "base/Model.h"

#include <cstddef> // for size_t
#include <memory>
#include <vector>

#include "base/Layer.h"
#include "base/LearningRule.h"
#include "networks/HopfieldLayer.h"

class Hopfield : public Model
{
public:
    Hopfield(size_t n)
        : Model(n)
    {
        layers.emplace_back(std::make_unique<HopfieldLayer>(
            std::make_shared<HebbianRule<float>>(),
            std::make_shared<StepPolarActivation<float>>(),
            n,
            n));
    }

    // Constructor with custom learning rule
    Hopfield(const std::shared_ptr<LearningRule<float>> &newRule,
             const std::shared_ptr<ActivationFunction<float>> &activationFunction,
             size_t n)
        : Model(n)
    {
        layers.emplace_back(
            std::make_unique<HopfieldLayer>(newRule, activationFunction, n, n));
    }

    ~Hopfield() override = default;

    void learn(const std::vector<Pattern> &patterns) override
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

    void learn(const std::vector<Pattern> &,
               const std::vector<Pattern> &,
               float,
               size_t) override
    {
        std::cerr << "Learn method not implemented for Hopfield network."
                  << std::endl;
        throw std::runtime_error(
            "Learn method not implemented for Hopfield network.");
    }

private:
};