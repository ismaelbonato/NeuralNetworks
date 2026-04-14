#pragma once
#include <cmath>
#include <stdexcept>
#include <limits>
#include <algorithm>

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

template<typename T>
class LogSigmoidActivation : public ActivationFunction<T>
{
public:
    inline T operator()(T x) const override
    {
        return std::log(static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x)));
        // Or equivalently: -std::log(1.0 + std::exp(-x))
    }
    inline T derivative(T x) const override
    {
        return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(x));
    }
};

template<typename T>
class LogActivation : public ActivationFunction<T>
{
public:
    inline T operator()(T x) const override
    {
        // Handle domain: log is only defined for positive values
        return (x > static_cast<T>(0)) ? std::log(x) : std::log(std::numeric_limits<T>::epsilon());
    }
    inline T derivative(T x) const override
    {
        return (x > static_cast<T>(0)) ? static_cast<T>(1.0) / x : static_cast<T>(0);
    }
};

template<typename T>
class SoftplusActivation : public ActivationFunction<T>
{
public:
    inline T operator()(T x) const override
    {
        return std::log(static_cast<T>(1.0) + std::exp(x));
    }
    inline T derivative(T x) const override
    {
        return static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x)); // sigmoid
    }
};

template<typename T>
class LogCoshActivation : public ActivationFunction<T>
{
public:
    inline T operator()(T x) const override
    {
        return std::log(std::cosh(x));
    }
    inline T derivative(T x) const override
    {
        return std::tanh(x);
    }
};


template<typename T>
class ScaledTanhActivation : public ActivationFunction<T>
{
public:
    inline T operator()(T x) const override
    {
        return (std::tanh(x) + static_cast<T>(1.0)) / static_cast<T>(2.0);
    }
    inline T derivative(T x) const override
    {
        T sech = static_cast<T>(1.0) / std::cosh(x);
        return static_cast<T>(0.5) * sech * sech;
    }
};


template<typename T>
class HardSigmoidActivation : public ActivationFunction<T>
{
public:
    inline T operator()(T x) const override
    {
        return std::max(static_cast<T>(0.0), 
               std::min(static_cast<T>(1.0), 
                       static_cast<T>(0.2) * x + static_cast<T>(0.5)));
    }
    inline T derivative(T x) const override
    {
        return (x > static_cast<T>(-2.5) && x < static_cast<T>(2.5)) ? 
               static_cast<T>(0.2) : static_cast<T>(0.0);
    }
};

template<typename T>
class SwishActivation : public ActivationFunction<T>
{
public:
    inline T operator()(T x) const override
    {
        return x / (static_cast<T>(1.0) + std::exp(-x)); // x * sigmoid(x)
    }
    inline T derivative(T x) const override
    {
        T sigmoid = static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
        return sigmoid * (static_cast<T>(1.0) + x * (static_cast<T>(1.0) - sigmoid));
    }
};

template<typename T>
class ScaledELUActivation : public ActivationFunction<T>
{
    T alpha = static_cast<T>(1.0);
public:
    inline T operator()(T x) const override
    {
        T elu = (x > static_cast<T>(0)) ? x : alpha * (std::exp(x) - static_cast<T>(1.0));
        // Scale to [0,1] - this is approximate
        return std::max(static_cast<T>(0.0), std::min(static_cast<T>(1.0), 
                       (elu + alpha) / (static_cast<T>(1.0) + alpha)));
    }
    inline T derivative(T x) const override
    {
        return (x > static_cast<T>(0)) ? static_cast<T>(1.0) : alpha * std::exp(x);
    }
};

template<typename T>
class MishActivation : public ActivationFunction<T>
{
public:
    inline T operator()(T x) const override
    {
        return x * std::tanh(std::log(static_cast<T>(1.0) + std::exp(x))); // x * tanh(softplus(x))
    }
    inline T derivative(T x) const override
    {
        T softplus = std::log(static_cast<T>(1.0) + std::exp(x));
        T tanh_sp = std::tanh(softplus);
        T sech_sp = static_cast<T>(1.0) / std::cosh(softplus);
        T sigmoid = static_cast<T>(1.0) / (static_cast<T>(1.0) + std::exp(-x));
        return tanh_sp + x * sech_sp * sech_sp * sigmoid;
    }
};


template<typename T>
class NegativeExpActivation : public ActivationFunction<T>
{
public:
    inline T operator()(T x) const override
    {
        return static_cast<T>(1.0) - std::exp(-x);
    }
    inline T derivative(T x) const override
    {
        return std::exp(-x);
    }
};