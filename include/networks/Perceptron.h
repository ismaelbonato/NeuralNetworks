#pragma once

#include "base/Model.h"
#include "networks/PerceptronLayer.h"
#include <memory>
#include <vector>

class Perceptron : public Model
{
public:
    Perceptron(const size_t in, const size_t out)
        : Model(in)
    {
        layers.emplace_back(std::make_unique<PerceptronLayer>(
            std::make_shared<PerceptronRule>(),
            std::make_shared<StepActivation<float>>(),
            in,
            out));
    }

    // Constructor with custom learning rule
    Perceptron(
        const std::shared_ptr<LearningRule> &newRule,
        const std::shared_ptr<ActivationFunction<float>> &activationFunction,
        size_t in,
        size_t out)
        : Model(in)
    {
        layers.emplace_back(std::make_unique<PerceptronLayer>(newRule,
                                                              activationFunction,
                                                              in,
                                                              out));
    }

    ~Perceptron() override = default;

    void learn(const std::vector<Pattern> &inputs,
               const std::vector<Pattern> &labels,
               float learningRate = 0.1f,
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