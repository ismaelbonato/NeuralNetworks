#pragma once

#include "base/Layer.h"
#include "base/Types.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace nn {

class Model
{
protected:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    Model();
    virtual ~Model();

    Layer &addLayer(std::unique_ptr<Layer> layer);

    Layer &getLayer(size_t index);
    const Layer &getLayer(size_t index) const;
    size_t numLayers() const;

    virtual Pattern infer(const Pattern &input);
};

} // namespace nn
