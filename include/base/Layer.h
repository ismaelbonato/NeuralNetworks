#pragma once
#include "Tensor.h"
#include "base/ActivationFunction.h"
#include "base/LearningRule.h"
#include "base/Types.h"

#include <memory>

class Layer
{
protected:
    size_t inputSize;
    size_t outputSize;
    Patterns weights;
    Pattern output;
    std::shared_ptr<LearningRule<Scalar>> learningRule;
    std::shared_ptr<ActivationFunction<Scalar>> activation;

    Pattern biases;

public:
    Layer() = delete;

    Layer(const std::shared_ptr<LearningRule<Scalar>> &newRule,
          const std::shared_ptr<ActivationFunction<Scalar>> &activationFunction,
          size_t in,
          size_t out)
        : inputSize(in)
        , outputSize(out)
        , learningRule(newRule)
        , activation(activationFunction)
    {}

    virtual ~Layer() = default;

    virtual std::unique_ptr<Layer> clone() const = 0;

    virtual void initWeights(Scalar value = 0.0f) = 0;

    size_t getInputSize() const { return inputSize; }
    size_t getOutputSize() const { return outputSize; }

    virtual void updateWeights(const Pattern &prev_activations,
                               const Pattern &delta,
                               Scalar learningRate)
    {

        auto gradMatrix = delta.outer(prev_activations);


        for (size_t i = 0; i < outputSize; ++i) {
            for (size_t j = 0; j < inputSize; ++j) {
                weights[i][j] = learningRule->updateWeight(weights[i][j],
                                                 gradMatrix[i][j],
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

        return weights.matVecMul(input) + biases;
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
        Pattern result(values.size());

        for (size_t i = 0; i < values.size(); ++i) {
            result[i] = (*activation)(values[i]);
        }
        return result;
    }

    Pattern backwardPass(const Pattern &delta, const Pattern &preActivation)
    {
        if (weights.empty() || weights.size() != outputSize) {
            throw std::runtime_error(
                "Weights are not initialized or size mismatch.");
        }
        return weights.matVecTransMul(delta) * activationDerivatives(preActivation);
    }
};