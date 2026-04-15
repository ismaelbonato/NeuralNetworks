#pragma once

#include "base/Types.h"
#include "networks/Perceptron.h"

#include <cstddef>
#include <memory>

namespace experimental
{

// Experimental evolutionary search over perceptron weights.
// Not intended as a stable training API.
class PerceptronNaturalSelection : public Perceptron
{
public:
    PerceptronNaturalSelection();
    PerceptronNaturalSelection(const std::shared_ptr<Layer> &newLayer);
    ~PerceptronNaturalSelection() override;

    void learn(const Batch &inputs,
               const Batch &labels,
               Scalar learningRate = 0.1f,
               size_t epochs = 100000) override;

    size_t findClosestPerceptron(const Batch &ret, const Batch &labels) const;
};

}
