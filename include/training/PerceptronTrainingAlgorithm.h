#pragma once

#include "base/Types.h"

#include <cstddef>

class Perceptron;

class PerceptronTrainingAlgorithm
{
public:
    virtual ~PerceptronTrainingAlgorithm() = default;

    virtual void learn(Perceptron &network,
                       const Batch &inputs,
                       const Batch &labels,
                       Scalar learningRate,
                       size_t epochs) = 0;
};
