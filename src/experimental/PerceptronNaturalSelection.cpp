#include "experimental/PerceptronNaturalSelection.h"

#include <limits>
#include <stdexcept>
#include <vector>

namespace experimental
{

PerceptronNaturalSelection::PerceptronNaturalSelection() = default;

PerceptronNaturalSelection::PerceptronNaturalSelection(const std::shared_ptr<Layer> &newLayer)
    : Perceptron(newLayer)
{}

PerceptronNaturalSelection::~PerceptronNaturalSelection() = default;

void PerceptronNaturalSelection::learn(const Batch &inputs,
                                       const Batch &labels,
                                       Scalar learningRate,
                                       size_t epochs)
{
    (void) learningRate;

    if (layers.empty()) {
        throw std::runtime_error("Cannot train natural-selection perceptron without a layer.");
    }

    auto layer = layers.front();
    std::vector<LayerParameters> candidates(4, layer->getParameters());

    size_t bestIdx = 0;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        Batch ret(4);

        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t candidateIdx = 0; candidateIdx < candidates.size(); ++candidateIdx) {
                layer->setParameters(candidates[candidateIdx]);
                ret[candidateIdx].push_back(layer->infer(inputs[i]).front());
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

size_t PerceptronNaturalSelection::findClosestPerceptron(const Batch &ret,
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

}
