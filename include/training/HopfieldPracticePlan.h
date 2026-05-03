#pragma once

#include "training/Coach.h"

class HopfieldPracticePlan : public PracticePlan
{
public:
    void practice(Model &network,
                  const PracticeData &data,
                  const PracticeOptions &options) const override;
};
