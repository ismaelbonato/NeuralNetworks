#include "base/Skill.h"

#include "layers/ConvolutionalLayer.h"
#include "layers/DenseLayer.h"
#include "layers/HopfieldLayer.h"

#include <functional>
#include <optional>
#include <typeinfo>

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

bool Skill::hasParameters() const
{
    return layerAs<const DenseLayer>(layer()).has_value()
           || layerAs<const ConvolutionalLayer>(layer()).has_value()
           || layerAs<const HopfieldLayer>(layer()).has_value();
}

std::optional<LayerParameters> Skill::parameters() const
{
    if (auto dense = layerAs<const DenseLayer>(layer())) {
        return dense->get().getParameters();
    }
    if (auto convolutional = layerAs<const ConvolutionalLayer>(layer())) {
        return convolutional->get().getParameters();
    }
    if (auto hopfield = layerAs<const HopfieldLayer>(layer())) {
        return hopfield->get().getParameters();
    }

    return std::nullopt;
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
        return;
    }
    if (auto convolutional = layerAs<ConvolutionalLayer>(layer())) {
        convolutional->get().setParameters(parameters);
        return;
    }
    if (auto hopfield = layerAs<HopfieldLayer>(layer())) {
        hopfield->get().setParameters(parameters);
        return;
    }

    throw std::runtime_error("Skill does not accept parameters.");
}

void Skill::requireInitialized() const
{
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
