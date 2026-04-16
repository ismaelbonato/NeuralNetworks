#pragma once

#include "base/Types.h"

#include <cstddef>

class Feedforward;

class FeedforwardTrainer
{
public:
    void learn(Feedforward &network,
               const Batch &inputs,
               const Batch &labels,
               Scalar learningRate,
               size_t epochs);
};
