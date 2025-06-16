#pragma once
#include "base/Layer.h"
#include "base/Model.h"
#include "networks/DenseLayer.h"
#include "base/Types.h"

class FeedforwardNetwork : public Model
{
protected:
    Patterns activate;       // Store activations for each layer
    Patterns preActivations; // Store pre-activations for each layer

public:

    // Default constructor creates a network with a single layer
    FeedforwardNetwork() = default; // No layers by default, can be extended later

    FeedforwardNetwork(const std::vector<size_t> &layerSizes)
        : activate(layerSizes.size())
        , preActivations(layerSizes.size() - 1)

    {
        for (size_t i = 1; i < layerSizes.size(); ++i) {
            layers.emplace_back(std::make_unique<DenseLayer>(
                std::make_shared<SGDRule<Scalar>>(),
                std::make_shared<SigmoidActivation<Scalar>>(),
                layerSizes[i - 1],
                layerSizes[i]));
        }
    }

    FeedforwardNetwork(
        const std::shared_ptr<LearningRule<Scalar>> &rule,
        const std::shared_ptr<ActivationFunction<Scalar>> &activationFunction,
        const std::vector<size_t> &layerSizes)
        : activate(layerSizes.size())
        , preActivations(layerSizes.size() - 1)
    {
        for (size_t i = 1; i < layerSizes.size(); ++i) {
            layers.emplace_back(
                std::make_unique<DenseLayer>(rule,
                                                   activationFunction,
                                                   layerSizes[i - 1],
                                                   layerSizes[i]));
        }
    }

    ~FeedforwardNetwork() override = default;

    void addLayer(std::unique_ptr<Layer> layer) override 
    {
        // Update the activate and preActivations vectors
        if (activate.empty()) {
            activate.emplace_back(Pattern(layer->getOutputSize(), 0.0f));
        } 
        activate.emplace_back(Pattern(layer->getOutputSize()));
        preActivations.emplace_back(Pattern(layer->getInputSize(), 0.0f));

        Model::addLayer(std::move(layer));
    }



    // Call the learning rule's learn method supervised
    void learn(const Patterns &inputs,
               const Patterns &labels,
               Scalar learningRate = 0.1f,
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

    void backpropagation(Pattern &delta, const Scalar rate)
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