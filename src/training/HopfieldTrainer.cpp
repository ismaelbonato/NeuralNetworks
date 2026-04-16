#include "training/HopfieldTrainer.h"

#include "networks/Hopfield.h"

#include <stdexcept>

void HopfieldTrainer::learn(Hopfield &network,
                            const Batch &inputs,
                            Scalar learningRate,
                            size_t epochs)
{
    (void) epochs;

    if (inputs.empty()) {
        throw std::runtime_error("Batch is empty.");
    }

    for (const auto &pattern : inputs) {
        for (auto &layer : network.getLayers()) {
            layer->updateWeights(pattern, {}, learningRate);
        }
    }
}
