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

    Pattern computeError(const Pattern &target, const Pattern &activated) const;
};
