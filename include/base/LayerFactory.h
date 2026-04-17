#pragma once

#include "base/Layer.h"

#include <memory>

template<typename LayerType>
std::unique_ptr<LayerType> makeLayer(const LayerConfig &config)
{
    auto layer = std::unique_ptr<LayerType>(new LayerType(config));
    layer->initializeParameters();
    return layer;
}
