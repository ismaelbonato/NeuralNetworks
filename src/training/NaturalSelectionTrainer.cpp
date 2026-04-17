#include "training/NaturalSelectionTrainer.h"

#include "base/Model.h"

#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
using ModelParameters = std::vector<LayerParameters>;

void validateTrainingData(const Model &network,
                          const Batch &inputs,
                          const Batch &labels)
{
    if (network.numLayers() == 0) {
        throw std::runtime_error("Cannot train natural-selection model without layers.");
    }
    if (inputs.empty() || inputs.size() != labels.size()) {
        throw std::runtime_error("Inputs and labels must be non-empty and have the same size.");
    }

    const size_t expectedInputSize = network.getLayers().front()->getInputSize();
    const size_t expectedOutputSize = network.getLayers().back()->getOutputSize();
    for (size_t sampleIndex = 0; sampleIndex < inputs.size(); ++sampleIndex) {
        if (inputs.at(sampleIndex).size() != expectedInputSize) {
            throw std::runtime_error("Training input size does not match model input size.");
        }
        if (labels.at(sampleIndex).size() != expectedOutputSize) {
            throw std::runtime_error("Training label size does not match model output size.");
        }
    }
}

ModelParameters snapshotParameters(const Model &network)
{
    ModelParameters parameters;
    parameters.reserve(network.numLayers());

    for (const auto &layer : network.getLayers()) {
        // cppcheck-suppress useStlAlgorithm
        parameters.push_back(layer->getParameters());
    }

    return parameters;
}

void applyParameters(Model &network, const ModelParameters &parameters)
{
    if (parameters.size() != network.numLayers()) {
        throw std::runtime_error("Candidate parameter count does not match model layers.");
    }

    for (size_t layerIndex = 0; layerIndex < network.numLayers(); ++layerIndex) {
        network.getLayer(layerIndex).setParameters(parameters.at(layerIndex));
    }
}

ModelParameters mutateParameters(const Model &network,
                                 const ModelParameters &parameters,
                                 Scalar mutationStrength)
{
    if (parameters.size() != network.numLayers()) {
        throw std::runtime_error("Candidate parameter count does not match model layers.");
    }

    ModelParameters mutatedParameters;
    mutatedParameters.reserve(parameters.size());

    for (size_t layerIndex = 0; layerIndex < network.numLayers(); ++layerIndex) {
        mutatedParameters.push_back(
            network.getLayer(layerIndex).naturalUpdatedParameters(parameters.at(layerIndex),
                                                                  mutationStrength));
    }

    return mutatedParameters;
}
}

NaturalSelectionTrainer::NaturalSelectionTrainer() = default;

NaturalSelectionTrainer::NaturalSelectionTrainer(NaturalSelectionConfig newConfig)
    : config(newConfig)
{}

void NaturalSelectionTrainer::learn(Model &network,
                                    const Batch &inputs,
                                    const Batch &labels,
                                    Scalar learningRate,
                                    size_t epochs)
{
    validateTrainingData(network, inputs, labels);
    if (config.populationSize == 0) {
        throw std::runtime_error("Natural-selection population size must be greater than zero.");
    }
    if (learningRate < Scalar{}) {
        throw std::runtime_error("Natural-selection mutation strength cannot be negative.");
    }

    const ModelParameters initialParameters = snapshotParameters(network);
    std::vector<ModelParameters> candidateParameters(config.populationSize, initialParameters);
    ModelParameters bestParameters = initialParameters;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        std::vector<Batch> candidatePredictions(candidateParameters.size());

        for (size_t candidateIndex = 0; candidateIndex < candidateParameters.size();
             ++candidateIndex) {
            applyParameters(network, candidateParameters.at(candidateIndex));

            for (size_t sampleIndex = 0; sampleIndex < inputs.size(); ++sampleIndex) {
                candidatePredictions.at(candidateIndex).push_back(
                    network.infer(inputs.at(sampleIndex)));
            }
        }

        const size_t bestCandidateIndex = findBestCandidate(candidatePredictions, labels);
        bestParameters = candidateParameters.at(bestCandidateIndex);

        std::vector<ModelParameters> nextGeneration(candidateParameters.size(), bestParameters);
        for (size_t candidateIndex = 1; candidateIndex < nextGeneration.size();
             ++candidateIndex) {
            nextGeneration.at(candidateIndex) = mutateParameters(network,
                                                              bestParameters,
                                                              learningRate);
        }

        candidateParameters = nextGeneration;
    }

    applyParameters(network, bestParameters);
}

size_t NaturalSelectionTrainer::findBestCandidate(
    const std::vector<Batch> &candidatePredictions,
    const Batch &labels) const
{
    if (candidatePredictions.empty()) {
        throw std::runtime_error("Candidate predictions cannot be empty.");
    }

    size_t bestCandidateIndex = 0;
    Scalar lowestSquaredError = std::numeric_limits<Scalar>::max();

    for (size_t candidateIndex = 0; candidateIndex < candidatePredictions.size();
         ++candidateIndex) {
        if (candidatePredictions.at(candidateIndex).size() != labels.size()) {
            throw std::runtime_error("Candidate prediction count does not match labels.");
        }

        Scalar totalSquaredError = 0.0f;
        for (size_t sampleIndex = 0; sampleIndex < candidatePredictions.at(candidateIndex).size();
             ++sampleIndex) {
            if (candidatePredictions.at(candidateIndex).at(sampleIndex).size()
                != labels.at(sampleIndex).size()) {
                throw std::runtime_error("Candidate prediction size does not match label size.");
            }

            const Pattern predictionError =
                candidatePredictions.at(candidateIndex).at(sampleIndex) - labels.at(sampleIndex);
            for (const Scalar value : predictionError) {
                // cppcheck-suppress useStlAlgorithm
                totalSquaredError += value * value;
            }
        }
        if (totalSquaredError < lowestSquaredError) {
            lowestSquaredError = totalSquaredError;
            bestCandidateIndex = candidateIndex;
        }
    }
    return bestCandidateIndex;
}
