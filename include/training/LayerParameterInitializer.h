#pragma once

#include "training/Initializer.h"

#include "base/LayerFactory.h"
#include "base/Skill.h"
#include "base/Types.h"

#include <memory>
#include <stdexcept>
#include <utility>

class Layer;
class Model;
class Skill;

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
void initializeSkillParameters(
    Skill &skill,
    const LayerParameterInitialization &initialization = {});
void initializeModelParameters(
    Model &network,
    const LayerParameterInitialization &initialization = {});

template<typename LayerType>
class TrainableSkill
{
public:
    explicit TrainableSkill(std::unique_ptr<LayerType> newLayer)
        : runtimeLayer(std::move(newLayer))
    {
        if (!runtimeLayer) {
            throw std::invalid_argument(
                "Cannot create a trainable skill without a layer.");
        }
    }

    TrainableSkill(const TrainableSkill &) = delete;
    TrainableSkill &operator=(const TrainableSkill &) = delete;
    TrainableSkill(TrainableSkill &&) noexcept = default;
    TrainableSkill &operator=(TrainableSkill &&) noexcept = default;
    ~TrainableSkill() = default;

    LayerType &layer()
    {
        return *runtimeLayer;
    }

    const LayerType &layer() const
    {
        return *runtimeLayer;
    }

    std::unique_ptr<LayerType> intoLayer()
    {
        if (!runtimeLayer) {
            throw std::runtime_error(
                "Cannot move a layer out of an empty trainable skill.");
        }
        return std::move(runtimeLayer);
    }

    Skill intoSkill()
    {
        return Skill(intoLayer());
    }

private:
    std::unique_ptr<LayerType> runtimeLayer;
};

template<typename LayerType, typename RecipeType>
TrainableSkill<LayerType> makeTrainableSkill(
    const RecipeType &recipe,
    const LayerParameterInitialization &initialization = {})
{
    auto layer = makeLayer<LayerType>(recipe);
    initializeLayerParameters(*layer, initialization);
    return TrainableSkill<LayerType>(std::move(layer));
}

template<typename LayerType, typename RecipeType>
std::unique_ptr<LayerType> makeInitializedLayer(
    const RecipeType &recipe,
    const LayerParameterInitialization &initialization = {})
{
    return makeTrainableSkill<LayerType>(recipe, initialization).intoLayer();
}
