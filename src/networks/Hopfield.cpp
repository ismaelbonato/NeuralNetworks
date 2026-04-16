#include "networks/Hopfield.h"

#include <stdexcept>

Hopfield::Hopfield() = default;

Hopfield::Hopfield(const std::shared_ptr<Layer> &newLayer)
    : Model({newLayer})
{}

Hopfield::~Hopfield() = default;

Layers &Hopfield::getLayers()
{
    return layers;
}

const Layers &Hopfield::getLayers() const
{
    return layers;
}

Pattern Hopfield::infer(const Pattern &input)
{
    if (layers.empty()) {
        throw std::runtime_error("No layers exist in the model to perform inference.");
    }
    return layers.front()->infer(input);
}
