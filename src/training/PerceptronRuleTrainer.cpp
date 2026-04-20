#include "training/PerceptronRuleTrainer.h"

#include "base/Model.h"

#include <stdexcept>

namespace
{
Pattern computeError(const Pattern &target, const Pattern &activated)
{
    return Pattern{target.front() - activated.front()};
}

TrainableLayer &requireTrainable(Layer &layer)
{
    auto *trainable = dynamic_cast<TrainableLayer *>(&layer);
    if (trainable == nullptr) {
        throw std::runtime_error("Perceptron training requires a trainable layer.");
    }

    return *trainable;
}
}

void PerceptronRuleTrainer::learn(Model &network,
                                  const Batch &inputs,
                                  const Batch &labels,
                                  Scalar learningRate,
                                  size_t epochs)
{
    if (network.numLayers() == 0) {
        throw std::runtime_error("Cannot train perceptron without a layer.");
    }

    auto &layer = requireTrainable(network.getLayer(0));
    if (layer.getOutputSize() != 1) {
        throw std::runtime_error("Perceptron supports exactly one output.");
    }
    if (inputs.empty() || inputs.size() != labels.size()) {
        throw std::runtime_error("Inputs and labels must be non-empty and have the same size.");
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            Pattern activated = network.infer(inputs.at(i));
            Pattern error = computeError(labels.at(i), activated);
            layer.updateWeights(inputs.at(i), error, learningRate);
        }
    }
}
