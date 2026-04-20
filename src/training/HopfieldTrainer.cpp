#include "training/HopfieldTrainer.h"

#include "base/Model.h"

#include <stdexcept>

namespace
{
TrainableLayer &requireTrainable(Layer &layer)
{
    auto *trainable = dynamic_cast<TrainableLayer *>(&layer);
    if (trainable == nullptr) {
        throw std::runtime_error("Hopfield training requires trainable layers.");
    }

    return *trainable;
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

    for (const auto &pattern : inputs) {
        for (size_t layerIndex = 0; layerIndex < network.numLayers(); ++layerIndex) {
            requireTrainable(network.getLayer(layerIndex)).updateWeights(pattern, {}, learningRate);
        }
    }
}
