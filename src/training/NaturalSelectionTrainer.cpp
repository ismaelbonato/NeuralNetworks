#include "training/NaturalSelectionTrainer.h"

#include "base/Model.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/DenseLayer.h"
#include "layers/HopfieldLayer.h"
#include "training/LayerParameterInitializer.h"

#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <typeinfo>
#include <vector>

namespace
{
using ModelParameters = std::vector<LayerParameters>;

template<typename LayerType>
std::optional<std::reference_wrapper<LayerType>> layerAs(Layer &layer)
{
    try {
        return std::ref(dynamic_cast<LayerType &>(layer));
    } catch (const std::bad_cast &) {
        return std::nullopt;
    }
}

template<typename LayerType>
std::optional<std::reference_wrapper<const LayerType>> layerAs(
    const Layer &layer)
{
    try {
        return std::cref(dynamic_cast<const LayerType &>(layer));
    } catch (const std::bad_cast &) {
        return std::nullopt;
    }
}

std::optional<LayerParameters> layerParameters(const Layer &layer)
{
    if (auto dense = layerAs<const DenseLayer>(layer)) {
        return dense->get().getParameters();
    }
    if (auto convolutional = layerAs<const ConvolutionalLayer>(layer)) {
        return convolutional->get().getParameters();
    }
    if (auto hopfield = layerAs<const HopfieldLayer>(layer)) {
        return hopfield->get().getParameters();
    }

    return std::nullopt;
}

void setLayerParameters(Layer &layer, const LayerParameters &parameters)
{
    if (auto dense = layerAs<DenseLayer>(layer)) {
        dense->get().setParameters(parameters);
    } else if (auto convolutional = layerAs<ConvolutionalLayer>(layer)) {
        convolutional->get().setParameters(parameters);
    } else if (auto hopfield = layerAs<HopfieldLayer>(layer)) {
        hopfield->get().setParameters(parameters);
    }
}

LayerParameters mutatedLayerParameters(const LayerParameters &parameters,
                                       Scalar mutationStrength)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Scalar> dis(-mutationStrength,
                                               mutationStrength);

    LayerParameters updated{
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
        throw std::runtime_error("Cannot train natural-selection model without layers.");
    }
    if (inputs.empty() || inputs.size() != labels.size()) {
        throw std::runtime_error("Inputs and labels must be non-empty and have the same size.");
    }

    const Shape &expectedInputShape = network.getLayer(0).getExpectedInputShape();
    const Shape &expectedOutputShape
        = network.getLayer(network.numLayers() - 1).getExpectedOutputShape();
    for (size_t sampleIndex = 0; sampleIndex < inputs.size(); ++sampleIndex) {
        if (!inputs.at(sampleIndex).hasShape(expectedInputShape)) {
            throw std::runtime_error("Training input shape does not match model input shape.");
        }
        if (!labels.at(sampleIndex).hasShape(expectedOutputShape)) {
            throw std::runtime_error("Training label shape does not match model output shape.");
        }
    }
}

ModelParameters snapshotParameters(const Model &network)
{
    ModelParameters parameters;
    parameters.reserve(network.numLayers());

    for (size_t layerIndex = 0; layerIndex < network.numLayers();
         ++layerIndex) {
        if (auto params = layerParameters(network.getLayer(layerIndex))) {
            parameters.push_back(*params);
        } else {
            parameters.push_back({});
        }
    }

    return parameters;
}

void applyParameters(Model &network, const ModelParameters &parameters)
{
    if (parameters.size() != network.numLayers()) {
        throw std::runtime_error("Candidate parameter count does not match model layers.");
    }

    for (size_t layerIndex = 0; layerIndex < network.numLayers(); ++layerIndex) {
        setLayerParameters(network.getLayer(layerIndex),
                           parameters.at(layerIndex));
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

    for (size_t layerIndex = 0; layerIndex < parameters.size(); ++layerIndex) {
        mutatedParameters.push_back(mutatedLayerParameters(
            parameters.at(layerIndex),
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

    initializeModelParameters(network);
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
            if (!candidatePredictions.at(candidateIndex).at(sampleIndex)
                     .hasShape(Shape(labels.at(sampleIndex).shape()))) {
                throw std::runtime_error("Candidate prediction shape does not match label shape.");
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
