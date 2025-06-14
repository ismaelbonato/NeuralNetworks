#pragma once

#include "base/Layer.h"
#include <stdexcept>
#include <vector>

class PerceptronLayer : public Layer
{
public:
    PerceptronLayer(size_t in, size_t out)
        : Layer(in, out)
    {
        weights.resize(out, Pattern(out, 0.0));
    }

    PerceptronLayer(const std::shared_ptr<LearningRule> &newRule,
                    size_t in,
                    size_t out)
        : Layer(newRule, in, out)
    {
        weights.resize(out, std::vector<float>(out, 0.0));
    }

    float activation(float value) const override
    {
        return value > 0 ? 1 : 0; // Classic perceptron uses step function
    }

    void learn(const Pattern &input,
               const Pattern &label,
               float learningRate = 0.1f) override
    {
        if (learningRule == nullptr) {
            throw std::runtime_error(
                "Learning rule is not set for this layer.");
        }

        for (size_t i = 0; i < outputSize; ++i) {
            // Compute weighted sum (no bias)
            float sum = 0.0f;
            for (size_t j = 0; j < inputSize; ++j) {
                sum += weights[i][j] * input[j];
            }
            // Apply activation
            float output = activation(sum);
            float error = label[i] - output;

            // Update weights (no bias update)
            for (size_t j = 0; j < inputSize; ++j) {
                learningRule->updateWeight(weights[i][j],
                                           learningRate * error * input[j],
                                           learningRate);
            }
        }
    }
};
