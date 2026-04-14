#pragma once

#include "Model.h"
#include "Types.h"

#include <cstddef>
#include <initializer_list>
#include <memory>

class LayeredModel : public Model
{
public:
    LayeredModel();
    LayeredModel(Layers &newLayers);
    LayeredModel(const std::initializer_list<std::shared_ptr<Layer>> &newLayers);
    ~LayeredModel() override;

    Layers::value_type &getLayer(size_t index);
    Layers &getLayers();
    const Layers &getLayers() const;
    size_t numLayers() const;

    virtual void addLayers(const std::initializer_list<std::shared_ptr<Layer>> &newLayers);
    virtual void removeLayer(size_t index);

    Pattern infer(const Pattern &input) override;
};
