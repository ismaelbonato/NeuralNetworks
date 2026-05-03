#pragma once

#include "base/Parameters.h"
#include "base/Types.h"
#include "training/LearningRule.h"

#include <memory>
#include <stdexcept>

class Model;

class Optimizer
{
public:
    Optimizer() = default;
    virtual ~Optimizer() = default;

    virtual Scalar update(Scalar value,
                          Scalar gradient,
                          Scalar learningRate) const = 0;
    virtual void step(Model &network,
                      const Batch &activations,
                      const Batch &layerDeltas,
                      Scalar learningRate) const = 0;
};

class LearningRuleOptimizer : public Optimizer
{
public:
    explicit LearningRuleOptimizer(
        std::shared_ptr<LearningRule<Scalar>> newLearningRule);

    Scalar update(Scalar value,
                  Scalar gradient,
                  Scalar learningRate) const override;
    void step(Model &network,
              const Batch &activations,
              const Batch &layerDeltas,
              Scalar learningRate) const override;

private:
    std::shared_ptr<LearningRule<Scalar>> learningRule;
    mutable Pattern weightGradientScratch;
    mutable Pattern biasGradientScratch;
    mutable Parameters parameterScratch;
};
