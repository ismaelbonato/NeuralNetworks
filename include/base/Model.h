#pragma once

#include "base/Layer.h"
#include "base/Skill.h"
#include "base/Types.h"

#include <cstddef>

class Model
{
protected:
    Skills skills;

public:
    Model();
    virtual ~Model();

    Skill &addSkill(Skill skill);

    Layer &getLayer(size_t index);
    const Layer &getLayer(size_t index) const;
    Skill &getSkill(size_t index);
    const Skill &getSkill(size_t index) const;
    size_t numLayers() const;

    virtual Pattern infer(const Pattern &input);
};
