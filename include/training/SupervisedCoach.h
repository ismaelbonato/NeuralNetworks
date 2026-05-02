#pragma once

#include "base/Types.h"

#include <cstddef>

class Model;

class SupervisedCoach
{
public:
    virtual ~SupervisedCoach() = default;

    virtual void learn(Model &network,
                       const Batch &inputs,
                       const Batch &labels,
                       Scalar learningRate,
                       size_t epochs) = 0;
};
