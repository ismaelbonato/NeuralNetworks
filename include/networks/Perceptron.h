#pragma once

#include "base/LayeredModel.h"
#include "base/Types.h"

#include <memory>

class Perceptron : public LayeredModel
{
public:
    Perceptron();
    Perceptron(const std::shared_ptr<Layer> &newLayer);
    ~Perceptron() override;

    void learn(const Patterns &inputs,
               const Patterns &labels,
               Scalar learningRate = 0.1f,
               size_t epochs = 10000) override;

    Pattern computeError(const Pattern &target, const Pattern &activated) const;
};
