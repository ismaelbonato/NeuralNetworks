#include "base/Model.h"

Model::Model() = default;

Model::Model(Layers &newLayers)
    : layers(newLayers)
{}

Model::Model(std::initializer_list<std::shared_ptr<Layer>> newLayers)
    : layers(newLayers)
{}

Model::~Model() = default;

void Model::addLayer(const std::shared_ptr<Layer> &layer)
{
    layers.push_back(layer);
}
