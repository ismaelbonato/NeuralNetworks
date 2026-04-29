#pragma once

#include "base/Layer.h"
#include "base/Skill.h"

#include <memory>

template<typename LayerType, typename RecipeType>
std::unique_ptr<LayerType> makeLayer(const RecipeType &recipe)
{
    return std::make_unique<LayerType>(recipe);
}

template<typename LayerType, typename RecipeType>
Skill makeSkill(const RecipeType &recipe)
{
    return Skill(makeLayer<LayerType>(recipe));
}
