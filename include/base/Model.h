#pragma once

#include "base/Layer.h"
#include "base/Types.h"

#include <cstddef>
#include <memory>

class Model
{
protected:
    Layers layers;

public:
    Model();
    virtual ~Model();

    Layer &addLayer(std::unique_ptr<Layer> layer);
    void removeLayer(size_t index);

    Layer &getLayer(size_t index);
    const Layer &getLayer(size_t index) const;
    const Layers &getLayers() const;
    size_t numLayers() const;

    virtual Pattern infer(const Pattern &input);
};
