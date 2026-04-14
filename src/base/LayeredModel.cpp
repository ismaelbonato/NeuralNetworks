#include "base/LayeredModel.h"

#include <stdexcept>
#include <vector>

LayeredModel::LayeredModel() = default;

LayeredModel::LayeredModel(Layers &newLayers)
    : Model(newLayers)
{}

LayeredModel::LayeredModel(const std::initializer_list<std::shared_ptr<Layer>> &newLayers)
    : Model(newLayers)
{}

LayeredModel::~LayeredModel() = default;

Layers::value_type &LayeredModel::getLayer(size_t index)
{
    if (index >= layers.size()) {
        throw std::out_of_range("Layer index out of range.");
    }
    return layers[index];
}

Layers &LayeredModel::getLayers()
{
    return layers;
}

const Layers &LayeredModel::getLayers() const
{
    return layers;
}

size_t LayeredModel::numLayers() const
{
    return layers.size();
}

void LayeredModel::addLayers(const std::initializer_list<std::shared_ptr<Layer>> &newLayers)
{
    for (auto &layer : newLayers) {
        layers.push_back(layer);
    }
}

void LayeredModel::removeLayer(size_t index)
{
    if (index < layers.size()) {
        layers.erase(layers.begin()
                     + static_cast<std::vector<std::shared_ptr<Layer>>::difference_type>(index));
    }
}

Pattern LayeredModel::infer(const Pattern &input)
{
    if (layers.empty()) {
        throw std::runtime_error("No layers exist in the model to perform inference.");
    }

    Pattern output = input;
    for (auto &layer : layers) {
        output = layer->infer(output);
    }
    return output;
}
