#pragma once

#include "base/Types.h"


#include <cmath>
#include <cstddef> // for size_t
#include <iostream>
#include <memory>
#include <stdexcept> // for std::runtime_error
#include <utility>   // for std::pair

template<typename T>
class LearningRule
{
protected:
public:
    LearningRule() = default;
    virtual ~LearningRule() = default;

    virtual T updateWeight(T weight, T gradient, T learningRate) const = 0;
};

template<typename T>
class HebbianRule : public LearningRule<T>
{
public:
    HebbianRule() = default;
    ~HebbianRule() override = default;

    T updateWeight(T weight, T gradient, T) const override
    {
        // Hebbian learning doesn't use gradient or learning rate in the traditional sense.
        // Instead, weights are updated based on the correlation of inputs.
        //Gradient here represents the correlation term.
        return weight + gradient;
    }
};

template<typename T>
class PerceptronRule : public LearningRule<T>
{
public:
    PerceptronRule() = default;
    ~PerceptronRule() override = default;

    T updateWeight(T weight, T gradient, T learningRate) const override
    {
        return weight + learningRate * gradient;
    }
};

template<typename T>
class SGDRule : public LearningRule<T>
{
public:
    SGDRule() = default;
    ~SGDRule() override = default;

    T updateWeight(T weight, T gradient, T learningRate) const override
    {
        return weight - learningRate * gradient;
    }
};

template<typename T>
class AdamRule : public LearningRule<T>
{
public:
    AdamRule() = default;
    ~AdamRule() override = default;

    T updateWeight(T, T, T) const override
    {
        throw std::runtime_error("AdamRule not implemented yet.");
    }
};
