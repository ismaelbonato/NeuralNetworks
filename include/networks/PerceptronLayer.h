#pragma once

#include "base/Layer.h"
#include <stdexcept>
#include <vector>
#include <random>

class PerceptronLayer : public Layer
{
public:
    PerceptronLayer() = delete; // Default constructor is not allowed
    PerceptronLayer(const std::shared_ptr<LearningRule> &newRule,
                    size_t in,
                    size_t out)
        : Layer(newRule, in, out)
    {
        initWeights();
    }
    ~PerceptronLayer() override = default;

    void initWeights(float value = 0.0f) override // each layer should initialize its weights
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0, 1.0);

        if (weights.empty()) {
            weights.resize(outputSize, Pattern(inputSize, value));

            for (auto &row : weights)
                for (auto &w : row)
                    w = dis(gen);
        }
        if (biases.empty()) {
            biases.resize(outputSize, value);

            for (auto &b : biases)
                b = dis(gen);
        }
    }

    float activation(float value) const override
    {
        return value > 0 ? 1 : 0; // Classic perceptron uses step function
    }

};