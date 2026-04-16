#pragma once

#include "training/PerceptronTrainingAlgorithm.h"

class PerceptronRuleTrainer : public PerceptronTrainingAlgorithm
{
public:
    void learn(Perceptron &network,
               const Batch &inputs,
               const Batch &labels,
               Scalar learningRate,
               size_t epochs) override;
};
