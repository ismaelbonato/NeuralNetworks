#pragma once

#include "training/Coach.h"

#include <vector>

struct NaturalSelectionConfig
{
    size_t populationSize = 4;
};

class NaturalSelectionPracticePlan : public PracticePlan
{
public:
    NaturalSelectionPracticePlan();
    explicit NaturalSelectionPracticePlan(NaturalSelectionConfig newConfig);

    void practice(Model &network,
                  const PracticeData &data,
                  const PracticeOptions &options) const override;

    size_t findBestCandidate(const std::vector<Batch> &candidatePredictions,
                             const Batch &labels) const;

private:
    NaturalSelectionConfig config;
};
