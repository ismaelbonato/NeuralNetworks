#pragma once

#include "base/Layer.h"
#include "base/Types.h"

#include <memory>
#include <optional>
#include <stdexcept>

class Skill
{
public:
    explicit Skill(std::unique_ptr<Layer> newLayer)
        : runtimeLayer(std::move(newLayer))
    {
        if (!runtimeLayer) {
            throw std::invalid_argument("Cannot create a skill without a layer.");
        }
    }

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

    Pattern perform(const Pattern &input) const
    {
        return runtimeLayer->infer(input);
    }

    bool hasParameters() const;
    std::optional<LayerParameters> parameters() const;
    LayerParameters getParameters() const;
    void setParameters(const LayerParameters &parameters);
    void requireInitialized() const;

    std::unique_ptr<Layer> intoLayer()
    {
        if (!runtimeLayer) {
            throw std::runtime_error("Cannot move a layer out of an empty skill.");
        }
        return std::move(runtimeLayer);
    }

private:
    std::unique_ptr<Layer> runtimeLayer;
};
