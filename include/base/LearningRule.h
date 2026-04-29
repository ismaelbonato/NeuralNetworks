#pragma once

template<typename T>
class LearningRule
{
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

    inline T updateWeight(T weight, T gradient, T) const override
    {
        return weight + gradient;
    }
};

template<typename T>
class PerceptronRule : public LearningRule<T>
{
public:
    PerceptronRule() = default;
    ~PerceptronRule() override = default;

    inline T updateWeight(T weight, T gradient, T learningRate) const override
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

    inline T updateWeight(T weight, T gradient, T learningRate) const override
    {
        return weight - learningRate * gradient;
    }
};
