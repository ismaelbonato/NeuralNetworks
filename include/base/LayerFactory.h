#pragma once

#include "base/Layer.h"

#include <memory>

template<typename LayerType, typename RecipeType>
std::unique_ptr<LayerType> makeLayer(const RecipeType &recipe)
{
    return std::make_unique<LayerType>(recipe);
}
