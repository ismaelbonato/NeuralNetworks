#pragma once
#include <cmath>
#include <stdexcept>

template<typename T>
class ActivationFunction
{
public:
    virtual ~ActivationFunction() = default;
    virtual T operator()(T x) const = 0;
    virtual T derivative(T x) const = 0;
};

template<typename T>
class SigmoidActivation : public ActivationFunction<T>
{
public:
    inline T operator()(T x) const override
    {
        return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
    }
    inline T derivative(T x) const override
    {
        T s = operator()(x);
        return s * (static_cast<T>(1.0) - s);
    }
};

template<typename T>
class StepActivation : public ActivationFunction<T>
{
public:
    inline T operator()(T x) const override
    {
        return (x >= 0) ? static_cast<T>(1.0) : static_cast<T>(0.0); // Step function
    }
    inline T derivative(T) const override
    {
        return static_cast<T>(0.0);
    }
};

template<typename T>
class StepPolarActivation : public ActivationFunction<T>
{
public:
    inline T operator()(T x) const override
    {
        return (x >= 0) ? static_cast<T>(1.0) : static_cast<T>(-1.0); // Step function
    }
    inline T derivative(T) const override
    {
        return static_cast<T>(0.0);
    }
};


// ReLU Activation
template<typename T>
class ReLUActivation : public ActivationFunction<T>
{
public:
    inline T operator()(T x) const override
    {
        return (x > static_cast<T>(0)) ? x : static_cast<T>(0);
    }
    inline T derivative(T x) const override
    {
        return (x > static_cast<T>(0)) ? static_cast<T>(1) : static_cast<T>(0);
    }
};

template<typename T>
class TanhActivation : public ActivationFunction<T>
{
public:
    inline T operator()(T x) const override
    {
        return std::tanh(x);
    }
    inline T derivative(T x) const override
    {
        T t = std::tanh(x);
        return static_cast<T>(1) - t * t;
    }
};