#include "training/HopfieldTrainer.h"

#include "base/Model.h"
#include "layers/HopfieldLayer.h"
#include "training/LayerParameterInitializer.h"
#include "training/Optimizer.h"

#include <memory>
#include <stdexcept>
#include <typeinfo>

namespace
{
HopfieldLayer &requireHopfieldLayer(Layer &layer)
{
    try {
        return dynamic_cast<HopfieldLayer &>(layer);
    } catch (const std::bad_cast &) {
        throw std::runtime_error("Hopfield training requires hopfield layers.");
    }
}
}

void HopfieldTrainer::learn(Model &network,
                            const Batch &inputs,
                            Scalar learningRate,
                            size_t epochs)
{
    (void) epochs;

    if (inputs.empty()) {
        throw std::runtime_error("Batch is empty.");
    }

    const LearningRuleOptimizer optimizer{
        std::make_shared<HebbianRule<Scalar>>()};
    initializeModelParameters(network);

    for (const auto &pattern : inputs) {
        const Batch activations(network.numLayers(), pattern);
        const Batch layerDeltas(network.numLayers());
        for (size_t layerIndex = 0; layerIndex < network.numLayers(); ++layerIndex) {
            requireHopfieldLayer(network.getLayer(layerIndex));
        }
        optimizer.step(network, activations, layerDeltas, learningRate);
    }
}
