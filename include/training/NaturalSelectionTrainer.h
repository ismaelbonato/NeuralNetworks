#pragma once

#include "training/SupervisedTrainer.h"

#include <vector>

struct NaturalSelectionConfig
{
    size_t populationSize = 4;
};

class NaturalSelectionTrainer : public SupervisedTrainer
{
public:
    NaturalSelectionTrainer();
    explicit NaturalSelectionTrainer(NaturalSelectionConfig newConfig);

    void learn(Model &network,
               const Batch &inputs,
               const Batch &labels,
               Scalar learningRate,
               size_t epochs) override;

    size_t findBestCandidate(const std::vector<Batch> &candidatePredictions,
                             const Batch &labels) const;

private:
    NaturalSelectionConfig config;
};
