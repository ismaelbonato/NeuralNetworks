#pragma once

#include <random>

template<typename T>
class Tensor;

template<typename T>
class Initializer
{
public:
    virtual ~Initializer() = default;
    virtual void fill(Tensor<T> &tensor) const = 0;
};

template<typename T>
class ZeroInitializer : public Initializer<T>
{
public:
    void fill(Tensor<T> &tensor) const override
    {
        tensor.generate([]() { return T{}; });
    }
};

template<typename T>
class ConstantInitializer : public Initializer<T>
{
public:
    explicit ConstantInitializer(T newValue)
        : value(newValue)
    {}

    void fill(Tensor<T> &tensor) const override
    {
        tensor.generate([this]() { return value; });
    }

private:
    T value;
};

template<typename T>
class UniformInitializer : public Initializer<T>
{
public:
    UniformInitializer(T newMin, T newMax)
        : min(newMin), max(newMax)
    {}

    void fill(Tensor<T> &tensor) const override
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(min, max);

        tensor.generate([&dis, &gen]() { return dis(gen); });
    }

private:
    T min;
    T max;
};
