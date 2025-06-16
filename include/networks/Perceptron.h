#pragma once

#include "base/Model.h"
#include "networks/DenseLayer.h"
#include "base/Types.h"

#include <memory>

class Perceptron : public Model
{
public:
    Perceptron(const size_t in, const size_t out)
    {
        layers.emplace_back(std::make_unique<DenseLayer>(
            std::make_shared<PerceptronRule<Scalar>>(),
            std::make_shared<StepActivation<Scalar>>(),
            in,
            out));
    }

    // Constructor with custom learning rule
    Perceptron(
        const std::shared_ptr<LearningRule<Scalar>> &newRule,
        const std::shared_ptr<ActivationFunction<Scalar>> &activationFunction,
        size_t in,
        size_t out)
    {
        layers.emplace_back(std::make_unique<DenseLayer>(newRule,
                                                              activationFunction,
                                                              in,
                                                              out));
    }

    ~Perceptron() override = default;

    void learn(const Patterns &inputs,
               const Patterns &labels,
               Scalar learningRate = 0.1f,
               size_t epochs = 10000) override
    {
        std::cout << "Training feedforward Network..." << std::endl;
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                Pattern activated = layers.front()->infer(inputs[i]);
                Pattern error = computeError(labels[i], activated);
                layers.front()->updateWeights(inputs[i], error, learningRate);
            }
        }
    }

    Pattern computeError(const Pattern &target, const Pattern &activated) const
    {
        return {target.front() - activated.front()};
    }
};