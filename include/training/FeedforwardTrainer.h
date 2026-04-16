#pragma once

#include "base/Types.h"

#include <cstddef>

class Model;

class FeedforwardTrainer
{
public:
    void learn(Model &network,
               const Batch &inputs,
               const Batch &labels,
               Scalar learningRate,
               size_t epochs);
};
