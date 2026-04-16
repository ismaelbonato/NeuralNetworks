#pragma once

#include "base/Types.h"

#include <cstddef>

class Model;

class HopfieldTrainer
{
public:
    void learn(Model &network,
               const Batch &inputs,
               Scalar learningRate = Scalar{1.0f},
               size_t epochs = 10000);
};
