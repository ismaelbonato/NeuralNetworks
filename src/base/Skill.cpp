#include "base/Skill.h"

#include <stdexcept>
#include <utility>

Skill::Skill(std::unique_ptr<Layer> newLayer)
    : runtimeLayer(std::move(newLayer))
{
    if (!runtimeLayer) {
        throw std::invalid_argument("Cannot create a skill without a layer.");
    }
    if (runtimeLayer->usesParameters()) {
        ownedParameters = Parameters{};
    }
}

Pattern Skill::perform(const Pattern &input) const
{
    if (ownedParameters) {
        return runtimeLayer->infer(input, *ownedParameters);
    }
    return runtimeLayer->infer(input);
}

bool Skill::hasParameters() const
{
    return runtimeLayer->usesParameters();
}

std::optional<Parameters> Skill::parameters() const
{
    if (!hasParameters()) {
        return std::nullopt;
    }

    return ownedParameters.value_or(Parameters{});
}

Parameters Skill::getParameters() const
{
    if (auto params = parameters()) {
        return *params;
    }

    throw std::runtime_error("Skill does not expose parameters.");
}

void Skill::setParameters(const Parameters &parameters)
{
    if (!hasParameters()) {
        throw std::runtime_error("Skill does not accept parameters.");
    }

    runtimeLayer->requireInitialized(parameters);
    ownedParameters = parameters;
}

void Skill::requireInitialized() const
{
    if (!hasParameters()) {
        return;
    }

    runtimeLayer->requireInitialized(getParameters());
}
