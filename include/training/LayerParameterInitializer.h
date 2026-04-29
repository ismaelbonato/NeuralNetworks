#pragma once

#include "training/Initializer.h"

#include "base/LayerFactory.h"
#include "base/Types.h"

#include <memory>

class Layer;
class Model;

struct LayerParameterInitialization
{
    std::shared_ptr<Initializer<Scalar>> weightInitializer
        = std::make_shared<UniformInitializer<Scalar>>(Scalar{-1.0},
                                                       Scalar{1.0});
    std::shared_ptr<Initializer<Scalar>> biasInitializer
        = std::make_shared<ConstantInitializer<Scalar>>(Scalar{0.0});
};

void initializeLayerParameters(
    Layer &layer,
    const LayerParameterInitialization &initialization = {});
void initializeModelParameters(
    Model &network,
    const LayerParameterInitialization &initialization = {});

template<typename LayerType, typename RecipeType>
std::unique_ptr<LayerType> makeInitializedLayer(
    const RecipeType &recipe,
    const LayerParameterInitialization &initialization = {})
{
    auto layer = makeLayer<LayerType>(recipe);
    initializeLayerParameters(*layer, initialization);
    return layer;
}
