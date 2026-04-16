#pragma once

#include "base/Layer.h"

#include <memory>

template<typename LayerType>
std::shared_ptr<LayerType> makeLayer(const LayerConfig &config)
{
    auto layer = std::shared_ptr<LayerType>(new LayerType(config));
    layer->initializeParameters();
    return layer;
}
