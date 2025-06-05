#ifndef PERCEPTRONLAYER_H
#define PERCEPTRONLAYER_H

#include "Layer.h"
#include <stdexcept>
#include <vector>

class PerceptronLayer : public Layer
{
public:
    PerceptronLayer(size_t in, size_t out)
        : Layer(in, out)
    {
        weights.resize(out, Pattern(out, 0.0));
    }

    PerceptronLayer(std::unique_ptr<LearningRule> newRule, size_t in, size_t out)
        : Layer(std::move(newRule), in, out)
    {
        weights.resize(out, std::vector<double>(out, 0.0));
    }

    int activation(double value) const override
    {
        return value > 0 ? 1 : 0; // Classic perceptron uses step function
    }

    Pattern forward(const Pattern &input) const override
    {
        Pattern result(outputSize, 0.0);
        for (size_t i = 0; i < outputSize; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < inputSize; ++j) {
                sum += weights[i][j] * input[j];
            }
            result[i] = activation(sum);
        }
        return result;
    }
};

#endif // PERCEPTRONLAYER_H