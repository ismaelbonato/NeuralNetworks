#pragma once

#include "base/Tensor.h"
#include "base/Types.h"

#include <cstddef>
#include <memory>

class GradientEngine;
class Model;
class Optimizer;

struct PracticeData
{
    Batch inputs;
    Batch labels;
};

struct PracticeOptions
{
    Scalar learningRate = Scalar{1.0F};
    size_t epochs = 1;
};

class PracticePlan
{
public:
    PracticePlan() = default;
    virtual ~PracticePlan() = default;

    virtual void practice(Model &network,
                          const PracticeData &data,
                          const PracticeOptions &options) const = 0;
};

class BackpropagationPracticePlan : public PracticePlan
{
public:
    BackpropagationPracticePlan();
    BackpropagationPracticePlan(std::unique_ptr<GradientEngine> newGradientEngine,
                                std::unique_ptr<Optimizer> newOptimizer);
    ~BackpropagationPracticePlan() override;

    void practice(Model &network,
                  const PracticeData &data,
                  const PracticeOptions &options) const override;

private:
    std::unique_ptr<GradientEngine> gradientEngine;
    std::unique_ptr<Optimizer> optimizer;
};

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
