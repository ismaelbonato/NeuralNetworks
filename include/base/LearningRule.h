#pragma once

#include <cmath>
#include <cstddef> // for size_t
#include <iostream>
#include <memory>
#include <stdexcept> // for std::runtime_error
#include <utility>   // for std::pair
#include <vector>

using Pattern = std::vector<float>;
using PairOfPatternAndBias = std::pair<std::vector<Pattern>, Pattern>;
class LearningRule
{
protected:
public:
    LearningRule() = default;
    virtual ~LearningRule() = default;

    virtual float updateWeight(float weight,
                               float gradient,
                               float learningRate) const
        = 0;
};

class HebbianRule : public LearningRule
{
public:
    HebbianRule() = default;
    ~HebbianRule() override = default;

    float updateWeight(float weight,
                       float gradient,
                       float) const override
    {
        // Hebbian learning doesn't use gradient or learning rate in the traditional sense.
        // Instead, weights are updated based on the correlation of inputs.
        return weight
               + gradient; // Gradient here represents the correlation term.
    }
};

class PerceptronRule : public LearningRule
{
public:
    PerceptronRule() = default;
    ~PerceptronRule() override = default;

    float updateWeight(float weight,
                       float gradient,
                       float learningRate) const override
    {
        // Perceptron update rule
        return weight + learningRate * gradient;
    }
};

class SGDRule : public LearningRule
{
public:
    SGDRule() = default;
    ~SGDRule() override = default;

    float updateWeight(float weight,
                       float gradient,
                       float learningRate) const override
    {
        return weight - learningRate * gradient;
    }
};

class AdamRule : public LearningRule
{
public:
    AdamRule() = default;
    ~AdamRule() override = default;

    float updateWeight(float,
                       float,
                       float) const override
    {
        throw std::runtime_error("AdamRule not implemented yet.");
    }
};
