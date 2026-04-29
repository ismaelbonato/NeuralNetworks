#include "base/Skill.h"

#include "layers/ConvolutionalLayer.h"
#include "layers/DenseLayer.h"
#include "layers/HopfieldLayer.h"

#include <functional>
#include <optional>
#include <typeinfo>
#include <utility>

namespace {
template<typename LayerType>
std::optional<std::reference_wrapper<LayerType>> layerAs(Layer &layer)
{
    try {
        return std::ref(dynamic_cast<LayerType &>(layer));
    } catch (const std::bad_cast &) {
        return std::nullopt;
    }
}

template<typename LayerType>
std::optional<std::reference_wrapper<const LayerType>> layerAs(
    const Layer &layer)
{
    try {
        return std::cref(dynamic_cast<const LayerType &>(layer));
    } catch (const std::bad_cast &) {
        return std::nullopt;
    }
}
} // namespace

Skill::Skill(std::unique_ptr<Layer> newLayer)
    : runtimeLayer(std::move(newLayer))
{
    if (!runtimeLayer) {
        throw std::invalid_argument("Cannot create a skill without a layer.");
    }
    adoptLayerParameters();
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
    return layerAs<const DenseLayer>(layer()).has_value()
           || layerAs<const ConvolutionalLayer>(layer()).has_value()
           || layerAs<const HopfieldLayer>(layer()).has_value();
}

std::optional<LayerParameters> Skill::parameters() const
{
    if (!hasParameters()) {
        return std::nullopt;
    }

    return ownedParameters.value_or(LayerParameters{});
}

LayerParameters Skill::getParameters() const
{
    if (auto params = parameters()) {
        return *params;
    }

    throw std::runtime_error("Skill does not expose parameters.");
}

void Skill::setParameters(const LayerParameters &parameters)
{
    if (auto dense = layerAs<DenseLayer>(layer())) {
        dense->get().setParameters(parameters);
        ownedParameters = parameters;
        return;
    }
    if (auto convolutional = layerAs<ConvolutionalLayer>(layer())) {
        convolutional->get().setParameters(parameters);
        ownedParameters = parameters;
        return;
    }
    if (auto hopfield = layerAs<HopfieldLayer>(layer())) {
        hopfield->get().setParameters(parameters);
        ownedParameters = parameters;
        return;
    }

    throw std::runtime_error("Skill does not accept parameters.");
}

void Skill::adoptLayerParameters()
{
    if (auto dense = layerAs<DenseLayer>(layer())) {
        ownedParameters = dense->get().getParameters();
        return;
    }
    if (auto convolutional = layerAs<ConvolutionalLayer>(layer())) {
        ownedParameters = convolutional->get().getParameters();
        return;
    }
    if (auto hopfield = layerAs<HopfieldLayer>(layer())) {
        ownedParameters = hopfield->get().getParameters();
        return;
    }

    ownedParameters = std::nullopt;
}

void Skill::requireInitialized() const
{
    syncParametersToLayer();
    if (auto dense = layerAs<const DenseLayer>(layer())) {
        dense->get().requireInitialized();
        return;
    }
    if (auto convolutional = layerAs<const ConvolutionalLayer>(layer())) {
        convolutional->get().requireInitialized();
        return;
    }
    if (auto hopfield = layerAs<const HopfieldLayer>(layer())) {
        hopfield->get().requireInitialized();
    }
}

void Skill::syncParametersToLayer() const
{
    if (!ownedParameters) {
        return;
    }

    if (auto dense = layerAs<DenseLayer>(*runtimeLayer)) {
        dense->get().setParameters(*ownedParameters);
        return;
    }
    if (auto convolutional = layerAs<ConvolutionalLayer>(*runtimeLayer)) {
        convolutional->get().setParameters(*ownedParameters);
        return;
    }
    if (auto hopfield = layerAs<HopfieldLayer>(*runtimeLayer)) {
        hopfield->get().setParameters(*ownedParameters);
    }
}
