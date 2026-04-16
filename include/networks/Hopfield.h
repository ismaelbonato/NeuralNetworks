#pragma once

#include "base/Layer.h"
#include "base/Model.h"
#include "base/Types.h"

#include <memory>

class Hopfield : public Model
{
public:
    Hopfield();
    Hopfield(const std::shared_ptr<Layer> &newLayer);
    ~Hopfield() override;

    Layers &getLayers();
    const Layers &getLayers() const;

    Pattern infer(const Pattern &input) override;
};
