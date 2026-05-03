#pragma once

#include "base/Tensor.h"
#include "base/Types.h"

#include <cstddef>
#include <memory>
#include <vector>

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
                          const PracticeOptions &options) const
        = 0;
};

class BackpropagationPracticePlan : public PracticePlan
{
public:
    BackpropagationPracticePlan();
    BackpropagationPracticePlan(std::unique_ptr<GradientEngine> newGradientEngine,
                                std::unique_ptr<Optimizer> newOptimizer);

    ~BackpropagationPracticePlan() override;

    BackpropagationPracticePlan(BackpropagationPracticePlan &&) noexcept;
    BackpropagationPracticePlan &operator=(
        BackpropagationPracticePlan &&) noexcept;

    BackpropagationPracticePlan(const BackpropagationPracticePlan &) = delete;
    BackpropagationPracticePlan &operator=(const BackpropagationPracticePlan &)
        = delete;

    void practice(Model &network,
                  const PracticeData &data,
                  const PracticeOptions &options) const override;

private:
    std::unique_ptr<GradientEngine> gradientEngine;
    std::unique_ptr<Optimizer> optimizer;
};

class PerceptronRulePracticePlan : public PracticePlan
{
public:
    void practice(Model &network,
                  const PracticeData &data,
                  const PracticeOptions &options) const override;
};

class HopfieldPracticePlan : public PracticePlan
{
public:
    void practice(Model &network,
                  const PracticeData &data,
                  const PracticeOptions &options) const override;
};

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
