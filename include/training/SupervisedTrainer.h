#pragma once

#include "base/Types.h"

#include <cstddef>

class Model;

class SupervisedTrainer
{
public:
    virtual ~SupervisedTrainer() = default;

    virtual void learn(Model &network,
                       const Batch &inputs,
                       const Batch &labels,
                       Scalar learningRate,
                       size_t epochs) = 0;
};
