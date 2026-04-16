#pragma once

#include "training/PerceptronTrainingAlgorithm.h"

class NaturalSelectionTrainer : public PerceptronTrainingAlgorithm
{
public:
    void learn(Perceptron &network,
               const Batch &inputs,
               const Batch &labels,
               Scalar learningRate,
               size_t epochs) override;

    size_t findClosestPerceptron(const Batch &ret, const Batch &labels) const;
};
