#pragma once

#include "training/SupervisedCoach.h"

class PerceptronRuleCoach : public SupervisedCoach
{
public:
    void learn(Model &network,
               const Batch &inputs,
               const Batch &labels,
               Scalar learningRate,
               size_t epochs) override;
};
