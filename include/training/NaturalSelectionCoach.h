#pragma once

#include "training/SupervisedCoach.h"

#include <vector>

struct NaturalSelectionConfig
{
    size_t populationSize = 4;
};

class NaturalSelectionCoach : public SupervisedCoach
{
public:
    NaturalSelectionCoach();
    explicit NaturalSelectionCoach(NaturalSelectionConfig newConfig);

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
