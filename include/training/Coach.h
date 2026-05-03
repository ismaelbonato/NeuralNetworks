#pragma once

#include "training/PracticePlan.h"

#include <memory>

class Coach
{
public:
    Coach();
    explicit Coach(std::unique_ptr<PracticePlan> newPracticePlan);
    ~Coach();

    void practice(Model &network,
                  const PracticeData &data,
                  const PracticeOptions &options) const;

private:
    std::unique_ptr<PracticePlan> practicePlan;
};
