#include "training/NaturalSelectionTrainer.h"

#include "base/Model.h"

#include <limits>
#include <stdexcept>
#include <vector>

void NaturalSelectionTrainer::learn(Model &network,
                                    const Batch &inputs,
                                    const Batch &labels,
                                    Scalar learningRate,
                                    size_t epochs)
{
    (void) learningRate;

    if (network.numLayers() == 0) {
        throw std::runtime_error("Cannot train natural-selection perceptron without a layer.");
    }
    if (inputs.empty() || inputs.size() != labels.size()) {
        throw std::runtime_error("Inputs and labels must be non-empty and have the same size.");
    }

    auto layer = network.getLayer(0);
    std::vector<LayerParameters> candidates(4, layer->getParameters());

    size_t bestIdx = 0;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        Batch ret(candidates.size());

        for (size_t candidateIdx = 0; candidateIdx < candidates.size(); ++candidateIdx) {
            layer->setParameters(candidates[candidateIdx]);

            for (size_t sampleIdx = 0; sampleIdx < inputs.size(); ++sampleIdx) {
                ret[candidateIdx].push_back(network.infer(inputs[sampleIdx]).front());
            }
        }

        bestIdx = findClosestPerceptron(ret, labels);

        const LayerParameters bestParameters = candidates[bestIdx];
        for (auto &candidate : candidates) {
            candidate = layer->naturalUpdatedParameters(bestParameters);
        }
    }

    layer->setParameters(layer->naturalUpdatedParameters(candidates[bestIdx]));
}

size_t NaturalSelectionTrainer::findClosestPerceptron(const Batch &ret,
                                                      const Batch &labels) const
{
    size_t bestIdx = 0;
    Scalar bestError = std::numeric_limits<Scalar>::max();

    for (size_t perceptronIdx = 0; perceptronIdx < ret.size(); ++perceptronIdx) {
        Scalar totalError = 0.0f;
        for (size_t sampleIdx = 0; sampleIdx < ret[perceptronIdx].size(); ++sampleIdx) {
            Scalar diff = ret[perceptronIdx][sampleIdx] - labels[sampleIdx].front();
            totalError += diff * diff;
        }
        if (totalError < bestError) {
            bestError = totalError;
            bestIdx = perceptronIdx;
        }
    }
    return bestIdx;
}
