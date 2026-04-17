#include "training/HopfieldTrainer.h"

#include "base/Model.h"

#include <stdexcept>

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
            network.getLayer(layerIndex).updateWeights(pattern, {}, learningRate);
        }
    }
}
