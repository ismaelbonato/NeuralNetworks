#include "networks/Hopfield.h"

#include <stdexcept>

Hopfield::Hopfield() = default;

Hopfield::Hopfield(const std::shared_ptr<Layer> &newLayer)
    : Model({newLayer})
{}

Hopfield::~Hopfield() = default;

void Hopfield::learn(const Patterns &inputs)
{
    learn(inputs, {}, 1.0f, 10000);
}

void Hopfield::learn(const Patterns &inputs,
                     const Patterns &labels,
                     Scalar learningRate,
                     size_t epochs)
{
    (void) epochs;
    (void) labels;

    if (inputs.empty()) {
        throw std::runtime_error("Patterns vector is empty.");
    }

    for (auto &pattern : inputs) {
        for (auto &layer : layers) {
            layer->updateWeights(pattern, {}, learningRate);
        }
    }
}

Pattern Hopfield::infer(const Pattern &input)
{
    if (layers.empty()) {
        throw std::runtime_error("No layers exist in the model to perform inference.");
    }
    return layers.front()->infer(input);
}
