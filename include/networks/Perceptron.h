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
        layers.emplace_back(
            std::make_unique<PerceptronLayer>(std::make_shared<PerceptronRule>(),
                                              in,
                                              out));
    }

    // Constructor with custom learning rule
    Perceptron(const std::shared_ptr<LearningRule> &newRule,
               size_t in,
               size_t out)
        : Model(in)
    {
        layers.emplace_back(std::make_unique<PerceptronLayer>(newRule, in, out));
    }

    ~Perceptron() override = default;

    void learn(const std::vector<Pattern> &inputs,
                       const std::vector<Pattern> &labels,
                       float learningRate = 0.1f,
                       size_t epochs = 100000) override
    {
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                for (auto &layer : layers) {
                    layer->learn(inputs[i], labels[i], learningRate);
                }
            }
        }
    }
};