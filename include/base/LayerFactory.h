#pragma once

#include "base/Layer.h"

#include <memory>

template<typename LayerType, typename RecipeType>
std::unique_ptr<LayerType> makeLayer(const RecipeType &recipe)
{
    auto layer = std::make_unique<LayerType>(recipe);
    if constexpr (requires { layer->initializeParameters(); }) {
        layer->initializeParameters();
    }
    return layer;
}
