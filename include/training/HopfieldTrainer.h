#pragma once

#include "base/Types.h"

#include <cstddef>

class Hopfield;

class HopfieldTrainer
{
public:
    void learn(Hopfield &network,
               const Batch &inputs,
               Scalar learningRate = Scalar{1.0f},
               size_t epochs = 10000);
};
