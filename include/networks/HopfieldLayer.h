#pragma once
#include "base/Layer.h"
#include "base/Types.h"

#include <algorithm> // for std::fill
#include <stdexcept>

class HopfieldLayer : public Layer
{
public:
    HopfieldLayer() = delete;

    HopfieldLayer(
        const std::shared_ptr<LearningRule<Scalar>> &newRule,
        const std::shared_ptr<ActivationFunction<Scalar>> &activationFunction,
        size_t in,
        size_t out)
        : Layer(newRule, activationFunction, in, out)
    {
        initWeights();
    }

    ~HopfieldLayer() override = default;

    virtual std::unique_ptr<Layer> clone() const override
    {
        return std::make_unique<HopfieldLayer>(
            learningRule, activation, inputSize, outputSize);
    }

    Pattern infer(const Pattern &input) const override
    {
        return recall(input); //  Return Value Optimization (RVO)
    }

    void updateWeights(const Pattern &pattern,
                               const Pattern &,
                               Scalar learningRate = Scalar{1.0f}) override
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

    void initWeights(Scalar value = Scalar{}) override
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
