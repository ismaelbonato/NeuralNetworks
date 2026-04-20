#pragma once

#include "base/Layer.h"

#include <memory>

template<typename LayerType, typename ConfigType>
std::unique_ptr<LayerType> makeLayer(const ConfigType &config)
{
    auto layer = std::unique_ptr<LayerType>(new LayerType(config));
    if constexpr (requires { layer->initializeParameters(); }) {
        layer->initializeParameters();
    }
    return layer;
}
