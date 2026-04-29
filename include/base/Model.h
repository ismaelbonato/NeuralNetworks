#pragma once

#include "base/Layer.h"
#include "base/Skill.h"
#include "base/Types.h"

#include <cstddef>
#include <functional>
#include <memory>

class Model
{
protected:
    Skills skills;

public:
    Model();
    virtual ~Model();

    Layer &addLayer(std::unique_ptr<Layer> layer);
    Skill &addSkill(Skill skill);
    void removeLayer(size_t index);

    Layer &getLayer(size_t index);
    const Layer &getLayer(size_t index) const;
    Skill &getSkill(size_t index);
    const Skill &getSkill(size_t index) const;
    const Skills &getSkills() const;
    std::vector<std::reference_wrapper<Layer>> getLayers();
    std::vector<std::reference_wrapper<const Layer>> getLayers() const;
    size_t numLayers() const;

    virtual Pattern infer(const Pattern &input);
};
