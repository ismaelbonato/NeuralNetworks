#include "training/PracticePlan.h"

#include "base/Model.h"
#include "base/Skill.h"
#include "training/ParameterInitializer.h"

#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using ModelParameters = std::vector<Parameters>;

Parameters mutatedParametersFor(const Parameters &parameters,
                                Scalar mutationStrength)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Scalar> dis(-mutationStrength,
                                               mutationStrength);

    Parameters updated{
        .weights = parameters.weights,
        .biases = parameters.biases,
    };

    if (!updated.weights.empty()) {
        updated.weights = updated.weights.map(
            [&dis, &gen](Scalar value) { return value + dis(gen); });
    }

    if (!updated.biases.empty()) {
        updated.biases = updated.biases.mapValues(
            [&dis, &gen](Scalar value) { return value + dis(gen); });
    }

    return updated;
}

void validateTrainingData(const Model &network,
                          const Batch &inputs,
                          const Batch &labels)
{
    if (network.numLayers() == 0) {
        throw std::runtime_error(
            "Cannot train natural-selection model without layers.");
    }
    if (inputs.empty() || inputs.size() != labels.size()) {
        throw std::runtime_error(
            "Inputs and labels must be non-empty and have the same size.");
    }

    const Shape &expectedInputShape
        = network.getLayer(0).getExpectedInputShape();
    const Shape &expectedOutputShape
        = network.getLayer(network.numLayers() - 1).getExpectedOutputShape();
    for (size_t sampleIndex = 0; sampleIndex < inputs.size(); ++sampleIndex) {
        if (!inputs.at(sampleIndex).hasShape(expectedInputShape)) {
            throw std::runtime_error(
                "Training input shape does not match model input shape.");
        }
        if (!labels.at(sampleIndex).hasShape(expectedOutputShape)) {
            throw std::runtime_error(
                "Training label shape does not match model output shape.");
        }
    }
}

ModelParameters snapshotParameters(const Model &network)
{
    ModelParameters parameters;
    parameters.reserve(network.numLayers());

    for (size_t layerIndex = 0; layerIndex < network.numLayers(); ++layerIndex) {
        const auto &skill = network.getSkill(layerIndex);
        if (skill.hasParameters()) {
            parameters.push_back(skill.getParameters());
        } else {
            parameters.push_back({});
        }
    }

    return parameters;
}

void applyParameters(Model &network, const ModelParameters &parameters)
{
    if (parameters.size() != network.numLayers()) {
        throw std::runtime_error(
            "Candidate parameter count does not match model layers.");
    }

    for (size_t layerIndex = 0; layerIndex < network.numLayers(); ++layerIndex) {
        if (network.getSkill(layerIndex).hasParameters()) {
            network.getSkill(layerIndex).setParameters(parameters.at(layerIndex));
        }
    }
}

ModelParameters mutateParameters(const Model &network,
                                 const ModelParameters &parameters,
                                 Scalar mutationStrength)
{
    if (parameters.size() != network.numLayers()) {
        throw std::runtime_error(
            "Candidate parameter count does not match model layers.");
    }

    ModelParameters mutatedParameters;
    mutatedParameters.reserve(parameters.size());

    for (size_t layerIndex = 0; layerIndex < parameters.size(); ++layerIndex) {
        mutatedParameters.push_back(
            mutatedParametersFor(parameters.at(layerIndex), mutationStrength));
    }

    return mutatedParameters;
}

size_t findBestCandidate(const std::vector<Batch> &candidatePredictions,
                         const Batch &labels)
{
    if (candidatePredictions.empty()) {
        throw std::runtime_error("Candidate predictions cannot be empty.");
    }

    size_t bestCandidateIndex = 0;
    Scalar lowestSquaredError = std::numeric_limits<Scalar>::max();

    for (size_t candidateIndex = 0;
         candidateIndex < candidatePredictions.size();
         ++candidateIndex) {
        if (candidatePredictions.at(candidateIndex).size() != labels.size()) {
            throw std::runtime_error(
                "Candidate prediction count does not match labels.");
        }

        Scalar totalSquaredError = 0.0f;
        for (size_t sampleIndex = 0;
             sampleIndex < candidatePredictions.at(candidateIndex).size();
             ++sampleIndex) {
            if (!candidatePredictions.at(candidateIndex)
                     .at(sampleIndex)
                     .hasShape(Shape(labels.at(sampleIndex).shape()))) {
                throw std::runtime_error(
                    "Candidate prediction shape does not match label shape.");
            }

            const Pattern predictionError
                = candidatePredictions.at(candidateIndex).at(sampleIndex)
                  - labels.at(sampleIndex);
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
} // namespace

NaturalSelectionPracticePlan::NaturalSelectionPracticePlan() = default;

NaturalSelectionPracticePlan::NaturalSelectionPracticePlan(
    NaturalSelectionConfig newConfig)
    : config(newConfig)
{}

void NaturalSelectionPracticePlan::practice(Model &network,
                                            const PracticeData &data,
                                            const PracticeOptions &options) const
{
    const auto &inputs = data.inputs;
    const auto &labels = data.labels;
    validateTrainingData(network, inputs, labels);
    if (config.populationSize == 0) {
        throw std::runtime_error(
            "Natural-selection population size must be greater than zero.");
    }
    if (options.learningRate < Scalar{}) {
        throw std::runtime_error(
            "Natural-selection mutation strength cannot be negative.");
    }

    initializeModelParameters(network);
    const ModelParameters initialParameters = snapshotParameters(network);
    std::vector<ModelParameters> candidateParameters(config.populationSize,
                                                     initialParameters);
    ModelParameters bestParameters = initialParameters;

    for (size_t epoch = 0; epoch < options.epochs; ++epoch) {
        std::vector<Batch> candidatePredictions(candidateParameters.size());

        for (size_t candidateIndex = 0;
             candidateIndex < candidateParameters.size();
             ++candidateIndex) {
            applyParameters(network, candidateParameters.at(candidateIndex));

            for (size_t sampleIndex = 0; sampleIndex < inputs.size();
                 ++sampleIndex) {
                candidatePredictions.at(candidateIndex)
                    .push_back(network.infer(inputs.at(sampleIndex)));
            }
        }

        const size_t bestCandidateIndex = findBestCandidate(candidatePredictions,
                                                            labels);
        bestParameters = candidateParameters.at(bestCandidateIndex);

        std::vector<ModelParameters> nextGeneration(candidateParameters.size(),
                                                    bestParameters);
        for (size_t candidateIndex = 1; candidateIndex < nextGeneration.size();
             ++candidateIndex) {
            nextGeneration.at(candidateIndex)
                = mutateParameters(network,
                                   bestParameters,
                                   options.learningRate);
        }

        candidateParameters = nextGeneration;
    }

    applyParameters(network, bestParameters);
}
