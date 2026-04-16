#pragma once

#include "training/SupervisedTrainer.h"

class NaturalSelectionTrainer : public SupervisedTrainer
{
public:
    void learn(Model &network,
               const Batch &inputs,
               const Batch &labels,
               Scalar learningRate,
               size_t epochs) override;

    size_t findClosestPerceptron(const Batch &ret, const Batch &labels) const;
};
