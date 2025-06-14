#pragma once
#include "base/Layer.h"

#include <stdexcept>
#include <vector>

class HopfieldLayer : public Layer
{
public:
    HopfieldLayer() = default;

    HopfieldLayer(size_t in, size_t out)
        : Layer(in, out)
    {
        weights.resize(in, std::vector<float>(out, 0.0));
    }

    HopfieldLayer(const std::shared_ptr<LearningRule> &newRule,
                  size_t in,
                  size_t out)
        : Layer(newRule, in, out)
    {
        weights.resize(in, std::vector<float>(out, 0.0));
    }

    ~HopfieldLayer() override = default;

    float activation(float value) const override
    {
        return (value >= 0) ? 1 : -1; // Bipolar activation function
    }

    Pattern infer(const Pattern &input) const override
    {
        return recall(input); //  Return Value Optimization (RVO)
    }

    // Apply the Hebbian learning rule to update weights
    void learn(const Pattern &pattern) override
    {
        size_t n = pattern.size();

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i != j) {
                    weights[i][j]
                        = learningRule->updateWeight(weights[i][j],
                                                        pattern[i] * pattern[j],
                                                        1.0f);
                }
            }
        }
    }


    // Overload infer: update until convergence
    Pattern recall(const Pattern &input) const
    {
        Pattern state = input;
        Pattern prev_state;
        do {
            prev_state = state;
            for (size_t i = 0; i < state.size(); ++i) {
                float sum = 0.0;
                for (size_t j = 0; j < state.size(); ++j) {
                    if (i != j) {
                        sum += learningRule->updateWeight(weights[i][j],
                                                          state[j],
                                                          1.0f);
                    }
                }
                state[i] = activation(sum);
            }
        } while (state != prev_state); // Repeat until state does not change
        return state;
    }
};
