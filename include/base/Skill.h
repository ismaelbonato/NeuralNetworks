#pragma once

#include "base/Layer.h"
#include "base/Types.h"

#include <memory>
#include <optional>

class Skill
{
public:
    explicit Skill(std::unique_ptr<Layer> newLayer);

    Skill(const Skill &) = delete;
    Skill &operator=(const Skill &) = delete;
    Skill(Skill &&) noexcept = default;
    Skill &operator=(Skill &&) noexcept = default;
    ~Skill() = default;

    Layer &layer()
    {
        return *runtimeLayer;
    }

    const Layer &layer() const
    {
        return *runtimeLayer;
    }

    Pattern perform(const Pattern &input) const;

    bool hasParameters() const;
    std::optional<Parameters> parameters() const;
    Parameters getParameters() const;
    void setParameters(const Parameters &parameters);
    void requireInitialized() const;

private:
    std::unique_ptr<Layer> runtimeLayer;
    std::optional<Parameters> ownedParameters;
};
