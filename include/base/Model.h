#pragma once

#include "base/Layer.h"
#include "base/Types.h"

#include <initializer_list>
#include <memory>

class Model
{
protected:
    Layers layers;

public:
    Model();
    Model(Layers &newLayers);
    Model(std::initializer_list<std::shared_ptr<Layer>> newLayers);
    virtual ~Model();

    virtual void addLayer(const std::shared_ptr<Layer> &layer);

    virtual Pattern infer(const Pattern &input) = 0;
};
