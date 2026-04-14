#pragma once

#include "base/Layer.h"
#include "base/Model.h"
#include "base/Types.h"

#include <cstddef>
#include <memory>

class Hopfield : public Model
{
public:
    Hopfield();
    Hopfield(const std::shared_ptr<Layer> &newLayer);
    ~Hopfield() override;

    void learn(const Patterns &inputs);
    void learn(const Patterns &inputs,
               const Patterns &labels,
               Scalar learningRate = 0.1f,
               size_t epochs = 100000) override;

    Pattern infer(const Pattern &input) override;
};
