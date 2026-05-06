#include "base/Model.h"

#include <stdexcept>
#include <utility>

namespace nn {

Model::Model() = default;

Model::~Model() = default;

Layer &Model::addLayer(std::unique_ptr<Layer> layer)
{
    if (!layer) {
        throw std::invalid_argument("Cannot add a null layer to the model.");
    }

    layers.push_back(std::move(layer));
    return *layers.back();
}

Layer &Model::getLayer(size_t index)
{
    if (index >= layers.size()) {
        throw std::out_of_range("Layer index out of range.");
    }
    return *layers.at(index);
}

const Layer &Model::getLayer(size_t index) const
{
    if (index >= layers.size()) {
        throw std::out_of_range("Layer index out of range.");
    }
    return *layers.at(index);
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
        // cppcheck-suppress useStlAlgorithm
        output = layer->infer(output);
    }
    return output;
}

} // namespace nn
