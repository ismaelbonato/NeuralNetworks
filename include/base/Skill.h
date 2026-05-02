#pragma once

#include "base/Layer.h"
#include "base/Types.h"

#include <memory>
#include <optional>
#include <stdexcept>

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
    std::optional<LayerParameters> parameters() const;
    LayerParameters getParameters() const;
    void setParameters(const LayerParameters &parameters);
    void requireInitialized() const;

private:
    std::unique_ptr<Layer> runtimeLayer;
    std::optional<LayerParameters> ownedParameters;
};
