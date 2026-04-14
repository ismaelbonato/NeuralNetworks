#pragma once

#include "Perceptron.h"
#include "base/Types.h"

#include <cstddef>
#include <memory>

class PerceptronNaturalSelection : public Perceptron
{
public:
    PerceptronNaturalSelection();
    PerceptronNaturalSelection(const std::shared_ptr<Layer> &newLayer);
    ~PerceptronNaturalSelection() override;

    void learn(const Patterns &inputs,
               const Patterns &labels,
               Scalar learningRate = 0.1f,
               size_t epochs = 100000) override;

    size_t findClosestPerceptron(const Patterns &ret, const Patterns &labels) const;
};
