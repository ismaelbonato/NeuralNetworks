#include "networks/Perceptron.h"

Perceptron::Perceptron() = default;

Perceptron::Perceptron(const std::shared_ptr<Layer> &newLayer)
    : LayeredModel({newLayer})
{}

Perceptron::~Perceptron() = default;

Pattern Perceptron::computeError(const Pattern &target, const Pattern &activated) const
{
    return Pattern{target.front() - activated.front()};
}
