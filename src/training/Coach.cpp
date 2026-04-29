#include "training/Coach.h"

#include "base/Layer.h"
#include "base/Model.h"
#include "base/Skill.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/DenseLayer.h"
#include "layers/HopfieldLayer.h"
#include "training/GradientEngine.h"
#include "training/LayerParameterInitializer.h"
#include "training/Optimizer.h"
#include "training/TrainingSession.h"

#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <typeinfo>
#include <utility>

namespace {
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

Pattern weightedInputFor(const Skill &skill, const Pattern &input)
{
    const auto &layer = skill.layer();
    if (auto dense = layerAs<const DenseLayer>(layer)) {
        skill.requireInitialized();
        if (input.empty()) {
            throw std::runtime_error("Input is empty");
        }

        const LayerParameters parameters = skill.getParameters();
        Pattern sums = input.matVec(parameters.weights);
        return parameters.biases.empty() ? sums : sums + parameters.biases;
    }
    if (auto convolutional = layerAs<const ConvolutionalLayer>(layer)) {
        skill.requireInitialized();
        if (input.empty()) {
            throw std::runtime_error("Input is empty");
        }

        const auto &recipe = convolutional->get().getConvolutionalRecipe();
        const LayerParameters parameters = skill.getParameters();
        Pattern result = input.conv1D(parameters.weights,
                                      recipe.stride,
                                      recipe.padding);
        if (!parameters.biases.empty()) {
            for (size_t outputChannel = 0;
                 outputChannel < recipe.outputChannels;
                 ++outputChannel) {
                for (size_t outputIndex = 0;
                     outputIndex < result.shape().at(1);
                     ++outputIndex) {
                    result.at({outputChannel, outputIndex})
                        += parameters.biases.at(outputChannel);
                }
            }
        }

        return result;
    }
    if (auto hopfield = layerAs<const HopfieldLayer>(layer)) {
        skill.requireInitialized();
        if (input.empty()) {
            throw std::runtime_error("Input is empty");
        }

        const LayerParameters parameters = skill.getParameters();
        Pattern sums = input.matVec(parameters.weights);
        return parameters.biases.empty() ? sums : sums + parameters.biases;
    }

    throw std::runtime_error(
        "Feedforward training requires weighted-input support.");
}

Pattern activateFor(const Layer &layer, const Pattern &values)
{
    const auto &activation = layer.getActivation();
    if (!activation) {
        throw std::runtime_error(
            "Activation function is not set for this layer.");
    }

    return values.map(
        [&activation](Scalar value) { return (*activation)(value); });
}

bool supportsParameterizedTraining(const Layer &layer)
{
    return layerAs<const DenseLayer>(layer).has_value()
           || layerAs<const ConvolutionalLayer>(layer).has_value()
           || layerAs<const HopfieldLayer>(layer).has_value();
}

void validateTrainingData(const Model &network,
                          const Batch &inputs,
                          const Batch &labels)
{
    if (network.numLayers() == 0) {
        throw std::runtime_error(
            "Cannot train feedforward network without layers.");
    }
    if (inputs.empty() || inputs.size() != labels.size()) {
        throw std::runtime_error(
            "Inputs and labels must be non-empty and have the same size.");
    }

    const auto &firstLayer = network.getLayer(0);
    const auto &lastLayer = network.getLayer(network.numLayers() - 1);
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs.at(i).hasShape(firstLayer.getInputShape())) {
            throw std::runtime_error(
                "Training input shape does not match network input shape.");
        }
        if (!labels.at(i).hasShape(lastLayer.getOutputShape())) {
            throw std::runtime_error(
                "Training label shape does not match network output shape.");
        }
    }
}

void forward(TrainingSession &session, const Pattern &input)
{
    auto &network = session.model();
    auto &activations = session.activations();
    auto &weightedInputs = session.weightedInputs();
    Pattern current = input;
    activations.at(0) = current;

    for (size_t layerIndex = 0; layerIndex < network.numLayers();
         ++layerIndex) {
        const auto &skill = network.getSkill(layerIndex);
        const auto &layer = skill.layer();
        if (supportsParameterizedTraining(layer)) {
            weightedInputs.at(layerIndex) = weightedInputFor(skill, current);
            current = activateFor(layer, weightedInputs.at(layerIndex));
        } else {
            current = layer.infer(current);
            weightedInputs.at(layerIndex) = current;
        }
        activations.at(layerIndex + 1) = current;
    }
}

Pattern lossDerivative(const Pattern &output, const Pattern &target)
{
    return output - target;
}
} // namespace

BackpropagationPracticePlan::BackpropagationPracticePlan()
    : BackpropagationPracticePlan(
          std::make_unique<BackpropagationGradientEngine>(),
          std::make_unique<LearningRuleOptimizer>(
              std::make_shared<SGDRule<Scalar>>()))
{}

BackpropagationPracticePlan::BackpropagationPracticePlan(
    std::unique_ptr<GradientEngine> newGradientEngine,
    std::unique_ptr<Optimizer> newOptimizer)
    : gradientEngine(std::move(newGradientEngine)),
      optimizer(std::move(newOptimizer))
{
    if (!gradientEngine) {
        throw std::invalid_argument(
            "Backpropagation practice requires a gradient engine.");
    }
    if (!optimizer) {
        throw std::invalid_argument(
            "Backpropagation practice requires an optimizer.");
    }
}

BackpropagationPracticePlan::~BackpropagationPracticePlan() = default;

void BackpropagationPracticePlan::practice(Model &network,
                                           const Batch &inputs,
                                           const Batch &labels,
                                           Scalar learningRate,
                                           size_t epochs) const
{
    validateTrainingData(network, inputs, labels);

    TrainingSession session(network);
    initializeModelParameters(network);
    session.initializeForwardBuffers();

    std::cout << "Training feedforward Network..." << std::endl;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t sampleIndex = 0; sampleIndex < inputs.size();
             ++sampleIndex) {
            forward(session, inputs.at(sampleIndex));
            const Pattern outputError = lossDerivative(
                session.activations().back(),
                labels.at(sampleIndex));
            session.setLayerDeltas(gradientEngine->computeLayerDeltas(
                network,
                session.activations(),
                session.weightedInputs(),
                outputError));

            optimizer->step(network,
                            session.activations(),
                            session.layerDeltas(),
                            learningRate);
        }
    }
}

Coach::Coach()
    : Coach(std::make_unique<BackpropagationPracticePlan>())
{}

Coach::Coach(std::unique_ptr<PracticePlan> newPracticePlan)
    : practicePlan(std::move(newPracticePlan))
{
    if (!practicePlan) {
        throw std::invalid_argument("Coach requires a practice plan.");
    }
}

Coach::~Coach() = default;

void Coach::practice(Model &network,
                     const Batch &inputs,
                     const Batch &labels,
                     Scalar learningRate,
                     size_t epochs) const
{
    practicePlan->practice(network, inputs, labels, learningRate, epochs);
}
