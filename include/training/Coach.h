#pragma once

#include "base/Types.h"

#include <cstddef>
#include <memory>

class GradientEngine;
class Model;
class Optimizer;

class PracticePlan
{
public:
    PracticePlan() = default;
    virtual ~PracticePlan() = default;

    virtual void practice(Model &network,
                          const Batch &inputs,
                          const Batch &labels,
                          Scalar learningRate,
                          size_t epochs) const = 0;
};

class BackpropagationPracticePlan : public PracticePlan
{
public:
    BackpropagationPracticePlan();
    BackpropagationPracticePlan(std::unique_ptr<GradientEngine> newGradientEngine,
                                std::unique_ptr<Optimizer> newOptimizer);
    ~BackpropagationPracticePlan() override;

    void practice(Model &network,
                  const Batch &inputs,
                  const Batch &labels,
                  Scalar learningRate,
                  size_t epochs) const override;

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
                  const Batch &inputs,
                  const Batch &labels,
                  Scalar learningRate,
                  size_t epochs) const;

private:
    std::unique_ptr<PracticePlan> practicePlan;
};
