#pragma once
#include "Tensor.h"
#include "base/ActivationFunction.h"
#include "base/LearningRule.h"
#include "base/Types.h"

#include <memory>

#include <ranges>

struct LayerConfig{
    std::shared_ptr<LearningRule<Scalar>> learningRule;
    std::shared_ptr<ActivationFunction<Scalar>> activation;
    size_t inputSize = 0;
    size_t outputSize = 0;
    std::string name;
    std::string type;
    std::string info;

    // Optional parameters with defaults
    bool useBias = true;
    Scalar weightInitScale = Scalar{1.0};
    Scalar biasInit = Scalar{0.0};

    // Validation
    inline bool isValid() const {
        return learningRule && activation && inputSize > 0 && outputSize > 0;
    }
};

class Layer
{
protected:
    LayerConfig config;
public:
    Patterns weights;
    Pattern biases;

public:
    Layer() = delete;

    Layer(const LayerConfig &newConfig)
        : config(newConfig)
    {
        if (!config.isValid()) {
            throw std::invalid_argument("Invalid layer configuration");
        }
        #pragma message("Do not forgot to deal with weights and bias initialization")
        //initWeights();
        if (config.useBias) {
            //initBiases(config.biasInit);
        }
    }

    virtual ~Layer() = default;

    virtual std::shared_ptr<Layer> clone() const = 0;
    virtual void initWeights(Scalar value = Scalar{}) = 0;

    inline size_t getInputSize() const { return config.inputSize; }
    inline size_t getOutputSize() const { return config.outputSize; }

    virtual void updateWeights(const Pattern &prev_activations,
                          const Pattern &delta,
                          Scalar learningRate)
    {
        // 
        // Direct loops - maximum performance
        for (size_t i = 0; i < config.outputSize; ++i) {
            for (size_t j = 0; j < config.inputSize; ++j) {
                // Compute gradient on-demand, no allocation
                Scalar gradient = delta[i] * prev_activations[j];
                weights[i][j] = config.learningRule->updateWeight(
                    weights[i][j], gradient, learningRate);
            }
        }

        // Direct bias updates
        for (size_t i = 0; i < config.outputSize; ++i) {
            biases[i] = config.learningRule->updateWeight(
                biases[i], delta[i], learningRate);
        }
    }

    virtual Pattern infer(const Pattern &input) const
    {
        if (input.size() != config.inputSize) {
            throw std::runtime_error(
                "Input size does not match layer input size.");
        }
        Pattern sums = weightedSum(input);
        return activate(sums);
    }

    virtual Pattern weightedSum(const Pattern &input) const
    {
        Pattern sums(config.outputSize, 0.0);

        if (input.empty()) {
            throw std::runtime_error("Input is empty");
        }

        return weights.matVecMul(input) + biases;
    }

    virtual Pattern activationDerivatives(const Pattern &values) const
    {
        Pattern result(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            result[i] = (*config.activation).derivative(values[i]);
        }
        return result;
    }

    virtual Pattern activate(const Pattern &values) const
    {
        if (config.learningRule == nullptr) {
            throw std::runtime_error(
                "Learning rule is not set for this layer.");
        }
        if (config.activation == nullptr) {
            throw std::runtime_error(
                "Activation function is not set for this layer.");
        }
        Pattern result(values.size());

        for (size_t i = 0; i < values.size(); ++i) {
            result[i] = (*config.activation)(values[i]);
        }
        return result;
    }

    Pattern backwardPass(const Pattern &delta, const Pattern &preActivation)
    {
        if (weights.empty() || weights.size() != config.outputSize) {
            throw std::runtime_error(
                "Weights are not initialized or size mismatch.");
        }
        return weights.matVecTransMul(delta) * activationDerivatives(preActivation);
    }
};