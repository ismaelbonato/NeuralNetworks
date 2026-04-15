#include "networks/Perceptron.h"

#include <stdexcept>

Perceptron::Perceptron() = default;

Perceptron::Perceptron(const std::shared_ptr<Layer> &newLayer)
    : LayeredModel({newLayer})
{}

Perceptron::~Perceptron() = default;

void Perceptron::learn(const Batch &inputs,
                       const Batch &labels,
                       Scalar learningRate,
                       size_t epochs)
{
    if (layers.empty()) {
        throw std::runtime_error("Cannot train perceptron without a layer.");
    }
    if (layers.front()->getOutputSize() != 1) {
        throw std::runtime_error("Perceptron supports exactly one output.");
    }
    if (inputs.empty() || inputs.size() != labels.size()) {
        throw std::runtime_error("Inputs and labels must be non-empty and have the same size.");
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            Pattern activated = layers.front()->infer(inputs[i]);
            Pattern error = computeError(labels[i], activated);
            layers.front()->updateWeights(inputs[i], error, learningRate);
        }
    }
}

Pattern Perceptron::computeError(const Pattern &target, const Pattern &activated) const
{
    return Pattern{target.front() - activated.front()};
}
