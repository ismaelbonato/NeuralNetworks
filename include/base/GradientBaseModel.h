#pragma once

#include "LayeredModel.h"

class GradientBaseModel : public LayeredModel
{
protected:
    Batch activate;
    Batch preActivations;

public:
    GradientBaseModel();
    GradientBaseModel(Layers &newlayers);
    GradientBaseModel(const std::initializer_list<std::shared_ptr<Layer>> &newlayers);
    ~GradientBaseModel() override;

    virtual void forward(const Pattern &input);
    virtual Pattern computeOutputError(const Pattern &target);
    virtual void backpropagation(const Pattern &outputError, Scalar rate);
    virtual Pattern lossDerivative(const Pattern &output, const Pattern &target);
};
