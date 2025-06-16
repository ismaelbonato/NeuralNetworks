#pragma once

#include "base/Layer.h" // Include the header file where Layer is defined
#include "base/LearningRule.h"
#include "base/Types.h" // Include for Pattern and Scalar types
#include <cstddef> // Include for size_t
#include <memory>
#include <vector>

class Model
{
protected:
    Layers layers;

public:

    Model() = default;
    virtual ~Model() = default;

    virtual void learn(const Patterns &inputs,
                       const Patterns &labels,
                       Scalar learningRate = 0.1f,
                       size_t epochs = 100000)
        = 0;

    virtual Pattern infer(const Pattern &input) = 0;

private:
    // You can add private members or methods if needed
};
