#pragma once

#include "base/GradientBaseModel.h"
#include "base/Layer.h"
#include "base/Types.h"

#include <initializer_list>
#include <memory>

class Feedforward : public GradientBaseModel
{
public:
    Feedforward();
    Feedforward(Layers &newlayers);
    Feedforward(const std::initializer_list<std::shared_ptr<Layer>> &newlayers);
    ~Feedforward() override;

    void learn(const Batch &inputs,
               const Batch &labels,
               Scalar learningRate = 0.1f,
               size_t epochs = 100000) override;
};
