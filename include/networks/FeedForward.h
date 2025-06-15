#pragma once
#include "base/Layer.h"
#include "base/Model.h"
#include "networks/FeedForwardLayer.h"

class FeedforwardNetwork : public Model
{
private:
    std::vector<Pattern> activate;       // Store activations for each layer
    std::vector<Pattern> preActivations; // Store pre-activations for each layer

public:
    FeedforwardNetwork(const std::vector<size_t> &layerSizes)
        : activate(layerSizes.size())
        , preActivations(layerSizes.size() - 1)

    {
        for (size_t i = 1; i < layerSizes.size(); ++i) {
            layers.emplace_back(std::make_unique<FeedforwardLayer>(
                std::make_shared<SGDRule>(),
                std::make_shared<SigmoidActivation<float>>(),
                layerSizes[i - 1],
                layerSizes[i]));
        }
    }

    FeedforwardNetwork(
        const std::shared_ptr<LearningRule> &rule,
        const std::shared_ptr<ActivationFunction<float>> &activationFunction,
        const std::vector<size_t> &layerSizes)
        : activate(layerSizes.size())
        , preActivations(layerSizes.size() - 1)
    {
        for (size_t i = 1; i < layerSizes.size(); ++i) {
            layers.emplace_back(
                std::make_unique<FeedforwardLayer>(rule,
                                                   activationFunction,
                                                   layerSizes[i - 1],
                                                   layerSizes[i]));
        }
    }

    ~FeedforwardNetwork() override = default;

    // Call the learning rule's learn method supervised
    void learn(const std::vector<Pattern> &inputs,
               const std::vector<Pattern> &labels,
               float learningRate = 0.1f,
               size_t epochs = 100000) override
    {
        std::cout << "Training feedforward Network..." << std::endl;
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                forward(inputs[i]);
                Pattern delta = computeError(labels[i]);
                backpropagation(delta, learningRate);
            }
        }
    }

    void forward(const Pattern &input)
    {
        // Forward pass: store activations and pre-activations
        Pattern current = input;
        activate[0] = current;
        for (size_t i = 0; i < layers.size(); i++) {
            preActivations[i] = layers[i]->weightedSum(current);
            current = layers[i]->activate(preActivations[i]);
            activate[i + 1] = current;
        }
    }

    Pattern computeError(const Pattern &target)
    {
        return elementwise_mul(lossDerivative(activate.back(), target),
                               layers.back()->activationDerivatives(
                                   preActivations.back()));
    }

    void backpropagation(Pattern &delta, const float rate)
    {
        // Backward pass: update weights and propagate error
        for (size_t l = layers.size(); l-- > 0;) {
            layers[l]->updateWeights(activate[l], delta, rate);
            if (l > 0) {
                delta = layers[l]->backwardPass(delta, preActivations[l - 1]);
            }
        }
    }
};