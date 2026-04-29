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
    void adoptLayerParameters();
    void requireInitialized() const;

    std::unique_ptr<Layer> intoLayer()
    {
        if (!runtimeLayer) {
            throw std::runtime_error("Cannot move a layer out of an empty skill.");
        }
        syncParametersToLayer();
        return std::move(runtimeLayer);
    }

private:
    std::unique_ptr<Layer> runtimeLayer;
    std::optional<LayerParameters> ownedParameters;

    void syncParametersToLayer() const;
};
