#pragma once

#include "base/Layer.h"
#include "base/Types.h"

#include <cstddef>
#include <initializer_list>
#include <memory>

class Model
{
protected:
    Layers layers;

public:
    Model();
    explicit Model(Layers &newLayers);
    explicit Model(const std::shared_ptr<Layer> &newLayer);
    Model(const std::initializer_list<std::shared_ptr<Layer>> &newLayers);
    virtual ~Model();

    virtual void addLayer(const std::shared_ptr<Layer> &layer);
    virtual void addLayers(const std::initializer_list<std::shared_ptr<Layer>> &newLayers);
    virtual void removeLayer(size_t index);

    Layers::value_type &getLayer(size_t index);
    const Layers::value_type &getLayer(size_t index) const;
    Layers &getLayers();
    const Layers &getLayers() const;
    size_t numLayers() const;

    virtual Pattern infer(const Pattern &input);
};
