#ifndef LAYER_H
#define LAYER_H

#include "LearningRule.h"
#include <memory>
#include <vector>

using Pattern = std::vector<double>;

class Layer
{
protected:
    size_t inputSize;
    size_t outputSize;
    std::vector<Pattern> weights;
    Pattern output;
    std::unique_ptr<LearningRule> learningRule;

public:
    Layer(size_t in, size_t out)
        : inputSize(in)
        , outputSize(out)
    {}

    Layer(std::unique_ptr<LearningRule> newRule, size_t in, size_t out)
        : inputSize(in)
        , outputSize(out)
        , learningRule(std::move(newRule))
    {}
    virtual ~Layer() = default;

    // Constructor to receive a unique_ptr and a size_t
    size_t getInputSize() const { return inputSize; }
    size_t getOutputSize() const { return outputSize; }

    virtual void learn(const std::vector<Pattern>& patterns)
    {
        auto w = learningRule->learn(patterns);
        setWeights(std::move(w));
    }

    virtual int activation(double value) const = 0;
    virtual Pattern forward(const Pattern &input) const = 0;
    //virtual void backward(const std::vector<Pattern>& patterns);
    virtual void setWeights(const std::vector<Pattern> &ws)
    {
        if (ws.size() != inputSize)
            throw std::invalid_argument("Weights size does not match input size.");

        this->weights = std::move(ws);
    }
};

#endif // LAYER_H