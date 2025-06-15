#pragma once
#include "base/Layer.h"

#include <algorithm> // for std::fill
#include <stdexcept>
#include <vector>

class HopfieldLayer : public Layer
{
public:
    HopfieldLayer() = delete;

    HopfieldLayer(
        const std::shared_ptr<LearningRule> &newRule,
        const std::shared_ptr<ActivationFunction<float>> &activationFunction,
        size_t in,
        size_t out)
        : Layer(newRule, activationFunction, in, out)
    {
        initWeights();
    }

    ~HopfieldLayer() override = default;

    Pattern infer(const Pattern &input) const override
    {
        return recall(input); //  Return Value Optimization (RVO)
    }

    virtual void updateWeights(const Pattern &pattern,
                               const Pattern &,
                               float learningRate = 1.0f)
    {
        size_t n = pattern.size();

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i != j) {
                    weights[i][j]
                        = learningRule->updateWeight(weights[i][j],
                                                        pattern[i] * pattern[j],
                                                        learningRate);
                }
            }
        }
    }

    void initWeights(float value = 0.0f) override
    {
        if (weights.empty()) {
            weights.resize(outputSize, Pattern(inputSize, value));
        }

        if (biases.empty()) {
            biases.resize(inputSize, value);
        }
    }

    // Overload infer: update until convergence
    Pattern recall(const Pattern &input) const
    {
        Pattern state = input;
        Pattern prev_state;
        do {
            prev_state = state;
            auto sum = weightedSum(input);
            state = activate(sum);
        } while (state != prev_state); // Repeat until state does not change
        return state;
    }
};
