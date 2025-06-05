#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include "Model.h"
#include "PerceptronLayer.h"
#include <memory>
#include <vector>

class Perceptron : public Model
{
public:
    Perceptron(const size_t in, const size_t out)
        : Model(in)
    {
        layers.emplace_back(
            std::make_unique<PerceptronLayer>(std::move(
                                                  std::make_unique<PerceptronRule>()),
                                              in,
                                              out));
    }

    // Constructor with custom learning rule
    Perceptron(std::unique_ptr<LearningRule> newRule, size_t in, size_t out)
        : Model(in)
    {
        layers.emplace_back(
            std::make_unique<PerceptronLayer>(std::move(newRule), in, out));
    }

    ~Perceptron() override = default;
};

#endif // PERCEPTRON_H