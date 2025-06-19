#pragma once

#include "base/Layer.h"
#include "base/Types.h"

#include <stdexcept>
#include <random>

class DenseLayer : public Layer
{
public:
    DenseLayer() = delete; // Default constructor is not allowed
    DenseLayer(const std::shared_ptr<LearningRule<Scalar>> &newRule,
        const std::shared_ptr<ActivationFunction<Scalar>> &activationFunction,
                    size_t in,
                    size_t out)
        : Layer(newRule, activationFunction, in, out)
    {
        initWeights();
    }
    ~DenseLayer() override = default;

    std::unique_ptr<Layer> clone() const override
    {
        return std::make_unique<DenseLayer>(
            learningRule, activation, inputSize, outputSize);
    }

    void initWeights(Scalar value = Scalar{}) override // each layer should initialize its weights
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<Scalar> dis(-1.0, 1.0);

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