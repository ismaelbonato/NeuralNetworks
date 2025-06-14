#pragma once
#include "base/LearningRule.h"
#include <memory>
#include <random>
#include <vector>

using Pattern = std::vector<float>;

class Layer
{
    //protected:
public:
    size_t inputSize;
    size_t outputSize;
    std::vector<Pattern> weights;
    Pattern output;
    std::shared_ptr<LearningRule> learningRule;
    Pattern biases;

public:
    Layer() = default;

    Layer(size_t in, size_t out)
        : inputSize(in)
        , outputSize(out)
    {
        initWeights();
    }

    Layer(const std::shared_ptr<LearningRule> &newRule, size_t in, size_t out)
        : inputSize(in)
        , outputSize(out)
        , learningRule(newRule)
    {
        initWeights();
    }

    virtual ~Layer() = default;

    virtual void learn(const Pattern &) 
    {
        std::cerr << "Learn method not implemented for this layer type." << std::endl;
        throw std::runtime_error("Learn method not implemented for this layer type.");
    }

    virtual void learn(const Pattern &, 
                       const Pattern &,
                       float)
    {
        std::cerr << "Learn method not implemented for this layer type." << std::endl;
        throw std::runtime_error("Learn method not implemented for this layer type.");
    }

    void initWeights()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0, 1.0);

        if (weights.empty()) {
            weights.resize(outputSize, Pattern(inputSize, 0.0));

            for (auto &row : weights)
                for (auto &w : row)
                    w = dis(gen);
        }
        if (biases.empty()) {
            biases.resize(outputSize, 0.0);

            for (auto &b : biases)
                b = dis(gen);
        }
    }

    size_t getInputSize() const { return inputSize; }
    size_t getOutputSize() const { return outputSize; }

    virtual void updateWeights(const Pattern &prev_activations,
                                const Pattern &delta,
                                float learningRate)
    {
        for (size_t i = 0; i < outputSize; ++i) {
            for (size_t j = 0; j < inputSize; ++j) {
                float grad = delta[i] * prev_activations[j];
                weights[i][j] = learningRule->updateWeight(weights[i][j],
                                                           grad,
                                                           learningRate);
            }
            biases[i] = learningRule->updateWeight(biases[i],
                                                   delta[i],
                                                   learningRate);
        }
    }

    virtual Pattern infer(const Pattern &input) const
    {
        if (input.size() != inputSize) {
            throw std::runtime_error(
                "Input size does not match layer input size.");
        }
        Pattern sums = weightedSum(input);
        return activate(sums);
    }

    virtual Pattern weightedSum(const Pattern &input) const
    {
        Pattern sums(outputSize, 0.0);

        if (input.empty()) {
            throw std::runtime_error("Input is empty");
        }

        float sum = 0.0;
        for (size_t i = 0; i < outputSize; ++i) {
            sum = biases[i];
            for (size_t j = 0; j < inputSize; ++j) {
                sum += weights[i][j] * input[j];
            }
            sums[i] = sum;
        }
        return sums;
    }

    virtual Pattern activate(const Pattern &values) const
    {
        if (learningRule == nullptr) {
            throw std::runtime_error(
                "Learning rule is not set for this layer.");
        }
        Pattern result(values.size(), 0.0);

        for (size_t i = 0; i < values.size(); ++i) {
            result[i] = activation(values[i]);
        }
        return result;
    }

    virtual float activation(float value) const = 0;
};