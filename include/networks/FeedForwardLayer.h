#pragma once
#include "base/Layer.h"
#include <random>

class FeedforwardLayer : public Layer
{
public:
    FeedforwardLayer(
        const std::shared_ptr<LearningRule<float>> &rule,
        const std::shared_ptr<ActivationFunction<float>> &activationFunction,
        size_t in,
        size_t out)
        : Layer(rule, activationFunction, in, out)
    {
        initWeights();
    }

    ~FeedforwardLayer() override = default;

    void initWeights(float value = 0.0f) override
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

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

};