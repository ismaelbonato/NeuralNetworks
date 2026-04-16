#pragma once

#include "training/SupervisedTrainer.h"

class PerceptronRuleTrainer : public SupervisedTrainer
{
public:
    void learn(Model &network,
               const Batch &inputs,
               const Batch &labels,
               Scalar learningRate,
               size_t epochs) override;
};
