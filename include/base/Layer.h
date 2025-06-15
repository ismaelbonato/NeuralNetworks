#pragma once
#include "base/ActivationFunction.h"
#include "base/LearningRule.h"
#include <memory>
#include <vector>

using Pattern = std::vector<float>;

// Matrix-vector multiplication: multiplies a matrix (vector of Pattern) by a Pattern vector
static Pattern matvec_mul(const std::vector<Pattern> &matrix, const Pattern &vec)
{
    if (matrix.empty() || matrix.size() != vec.size())
        throw std::runtime_error(
            "Matrix and vector size mismatch in matvec_mul.");
    auto size = matrix.at(0).size();
    Pattern result(size, 0.0);

    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += matrix[j][i] * vec[j];
        }
    }

    return result;
}

// Element-wise multiplication of two Pattern vectors
static Pattern elementwise_mul(const Pattern &a, const Pattern &b)
{
    if (a.size() != b.size())
        throw std::runtime_error("Size mismatch in elementwise_mul.");
    Pattern result(a.size());

    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }

    return result;
}

class Layer
{
    //protected:
public:
    size_t inputSize;
    size_t outputSize;
    std::vector<Pattern> weights;
    Pattern output;
    std::shared_ptr<LearningRule<float>> learningRule;
    std::shared_ptr<ActivationFunction<float>> activation;

    Pattern biases;

public:
    Layer() = delete;

    Layer(const std::shared_ptr<LearningRule<float>> &newRule,
          const std::shared_ptr<ActivationFunction<float>> &activationFunction,
          size_t in,
          size_t out)
        : inputSize(in)
        , outputSize(out)
        , learningRule(newRule)
        , activation(activationFunction)
    {}

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

    virtual Pattern activationDerivatives(const Pattern &values) const
    {
        Pattern result(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            result[i] = (*activation).derivative(values[i]);
        }
        return result;
    }

    virtual Pattern activate(const Pattern &values) const
    {
        if (learningRule == nullptr) {
            throw std::runtime_error(
                "Learning rule is not set for this layer.");
        }
        if (activation == nullptr) {
            throw std::runtime_error(
                "Activation function is not set for this layer.");
        }
        Pattern result(values.size(), 0.0);

        for (size_t i = 0; i < values.size(); ++i) {
            result[i] = (*activation)(values[i]);
        }
        return result;
    }

    Pattern backwardPass(const Pattern &delta,
                         const Pattern &preActivation)
    {
        if (weights.empty() || weights.size() != outputSize) {
            throw std::runtime_error(
                "Weights are not initialized or size mismatch.");
        }
        
        auto multiplyResult = matvec_mul(weights, delta);
        auto newPreActivation = activationDerivatives(preActivation);

        return elementwise_mul(multiplyResult, newPreActivation);
    }

};