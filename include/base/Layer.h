#pragma once
#include "base/LearningRule.h"
#include <memory>
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
    Layer() = delete;

    Layer(const std::shared_ptr<LearningRule> &newRule, size_t in, size_t out)
        : inputSize(in)
        , outputSize(out)
        , learningRule(newRule)
    {
    }

    virtual ~Layer() = default;

    virtual void initWeights(float value = 0.0f) = 0;

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