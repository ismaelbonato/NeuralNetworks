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

struct ParameterInitialization
{
    std::shared_ptr<Initializer<Scalar>> weightInitializer
        = std::make_shared<UniformInitializer<Scalar>>(Scalar{-1.0},
                                                       Scalar{1.0});
    std::shared_ptr<Initializer<Scalar>> biasInitializer
        = std::make_shared<ConstantInitializer<Scalar>>(Scalar{0.0});
};

void initializeSkillParameters(
    Skill &skill,
    const ParameterInitialization &initialization = {});
void initializeModelParameters(
    Model &network,
    const ParameterInitialization &initialization = {});

template<typename LayerType>
class TrainableSkill
{
public:
    explicit TrainableSkill(Skill newSkill)
        : runtimeSkill(std::move(newSkill))
    {}

    TrainableSkill(const TrainableSkill &) = delete;
    TrainableSkill &operator=(const TrainableSkill &) = delete;
    TrainableSkill(TrainableSkill &&) noexcept = default;
    TrainableSkill &operator=(TrainableSkill &&) noexcept = default;
    ~TrainableSkill() = default;

    LayerType &layer()
    {
        return dynamic_cast<LayerType &>(runtimeSkill.layer());
    }

    const LayerType &layer() const
    {
        return dynamic_cast<const LayerType &>(runtimeSkill.layer());
    }

    Skill intoSkill()
    {
        return std::move(runtimeSkill);
    }

private:
    Skill runtimeSkill;
};

template<typename LayerType, typename RecipeType>
TrainableSkill<LayerType> makeTrainableSkill(
    const RecipeType &recipe,
    const ParameterInitialization &initialization = {})
{
    Skill skill(makeLayer<LayerType>(recipe));
    initializeSkillParameters(skill, initialization);
    return TrainableSkill<LayerType>(std::move(skill));
}
