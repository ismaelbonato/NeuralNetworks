#include "training/PerceptronRuleTrainer.h"

#include "base/Model.h"
#include "layers/DenseLayer.h"
#include "training/ParameterInitializer.h"
#include "training/Optimizer.h"

#include <memory>
#include <stdexcept>
#include <typeinfo>

namespace
{
Pattern computeError(const Pattern &target, const Pattern &activated)
{
    return Pattern{target.front() - activated.front()};
}

DenseLayer &requireDenseLayer(Layer &layer)
{
    try {
        return dynamic_cast<DenseLayer &>(layer);
    } catch (const std::bad_cast &) {
        throw std::runtime_error("Perceptron training requires a dense layer.");
    }
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

    auto &layer = requireDenseLayer(network.getLayer(0));
    if (layer.getOutputSize() != 1) {
        throw std::runtime_error("Perceptron supports exactly one output.");
    }
    if (inputs.empty() || inputs.size() != labels.size()) {
        throw std::runtime_error("Inputs and labels must be non-empty and have the same size.");
    }

    const LearningRuleOptimizer optimizer{
        std::make_shared<PerceptronRule<Scalar>>()};
    initializeModelParameters(network);

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            Pattern activated = network.infer(inputs.at(i));
            Pattern error = computeError(labels.at(i), activated);
            optimizer.step(network, {inputs.at(i)}, {error}, learningRate);
        }
    }
}
