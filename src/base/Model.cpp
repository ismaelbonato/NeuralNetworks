#include "base/Model.h"

#include <stdexcept>
#include <vector>

Model::Model() = default;

Model::Model(Layers &newLayers)
    : layers(newLayers)
{}

Model::Model(const std::shared_ptr<Layer> &newLayer)
    : layers({newLayer})
{}

Model::Model(const std::initializer_list<std::shared_ptr<Layer>> &newLayers)
    : layers(newLayers)
{}

Model::~Model() = default;

void Model::addLayer(const std::shared_ptr<Layer> &layer)
{
    layers.push_back(layer);
}

void Model::addLayers(const std::initializer_list<std::shared_ptr<Layer>> &newLayers)
{
    layers.insert(layers.end(), newLayers.begin(), newLayers.end());
}

void Model::removeLayer(size_t index)
{
    if (index < layers.size()) {
        layers.erase(layers.begin()
                     + static_cast<std::vector<std::shared_ptr<Layer>>::difference_type>(index));
    }
}

Layers::value_type &Model::getLayer(size_t index)
{
    if (index >= layers.size()) {
        throw std::out_of_range("Layer index out of range.");
    }
    return layers[index];
}

Layers &Model::getLayers()
{
    return layers;
}

const Layers &Model::getLayers() const
{
    return layers;
}

size_t Model::numLayers() const
{
    return layers.size();
}

Pattern Model::infer(const Pattern &input)
{
    if (layers.empty()) {
        throw std::runtime_error("No layers exist in the model to perform inference.");
    }

    Pattern output = input;
    for (const auto &layer : layers) {
        output = layer->infer(output);
    }
    return output;
}
