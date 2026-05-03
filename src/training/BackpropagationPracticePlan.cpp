#include "training/PracticePlan.h"

#include "base/Layer.h"
#include "base/Model.h"
#include "base/Skill.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/DenseLayer.h"
#include "layers/FlattenLayer.h"
#include "layers/HopfieldLayer.h"
#include "training/GradientEngine.h"
#include "training/ParameterInitializer.h"
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
void fillFlattenOutput(const Pattern &input, Pattern &output)
{
    if (input.size() != output.size()) {
        throw std::runtime_error("Flatten output size does not match input size.");
    }

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i];
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

template<typename LayerType>
void fillMatrixPreActivation(const LayerType &layer,
                             const Parameters &parameters,
                             const Pattern &input,
                             Pattern &output)
{
    if (!output.hasShape(layer.getExpectedOutputShape())) {
        throw std::runtime_error("Dense pre-activation buffer shape mismatch.");
    }
    if (!input.hasShape(layer.getExpectedInputShape())) {
        throw std::runtime_error("Dense input shape mismatch.");
    }

    for (size_t row = 0; row < output.size(); ++row) {
        Scalar sum = parameters.biases.empty() ? Scalar{} : parameters.biases.at(row);
        for (size_t col = 0; col < input.size(); ++col) {
            sum += parameters.weights.at({row, col}) * input.at(col);
        }
        output.at(row) = sum;
    }
}

void fillConvolutionalPreActivation(const ConvolutionalLayer &layer,
                                    const Parameters &parameters,
                                    const Pattern &input,
                                    Pattern &output)
{
    if (!output.hasShape(layer.getExpectedOutputShape())) {
        throw std::runtime_error(
            "Convolutional pre-activation buffer shape mismatch.");
    }
    if (!input.hasShape(layer.getExpectedInputShape())) {
        throw std::runtime_error("Convolutional input shape mismatch.");
    }

    const auto &recipe = layer.getConvolutionalRecipe();
    for (size_t outputChannel = 0; outputChannel < recipe.outputChannels;
         ++outputChannel) {
        for (size_t outputIndex = 0; outputIndex < output.shape().at(1);
             ++outputIndex) {
            Scalar sum
                = parameters.biases.empty() ? Scalar{} : parameters.biases.at(outputChannel);
            for (size_t inputChannel = 0; inputChannel < recipe.inputChannels;
                 ++inputChannel) {
                for (size_t kernelIndex = 0; kernelIndex < recipe.kernelSize;
                     ++kernelIndex) {
                    const size_t paddedInputIndex
                        = (outputIndex * recipe.stride) + kernelIndex;

                    if (paddedInputIndex < recipe.padding) {
                        continue;
                    }

                    const size_t inputIndex = paddedInputIndex - recipe.padding;
                    if (inputIndex >= recipe.inputLength) {
                        continue;
                    }

                    sum += input.at({inputChannel, inputIndex})
                           * parameters.weights.at(
                               {outputChannel, inputChannel, kernelIndex});
                }
            }
            output.at({outputChannel, outputIndex}) = sum;
        }
    }
}

void fillParameterizedPreActivation(const Skill &skill,
                                    const Pattern &input,
                                    Pattern &output)
{
    skill.requireInitialized();
    if (input.empty()) {
        throw std::runtime_error("Input is empty");
    }

    const auto &layer = skill.layer();
    const auto &parameters = skill.getParameters();
    if (auto dense = layerAs<const DenseLayer>(layer)) {
        fillMatrixPreActivation(dense->get(), parameters, input, output);
        return;
    }
    if (auto convolutional = layerAs<const ConvolutionalLayer>(layer)) {
        fillConvolutionalPreActivation(convolutional->get(),
                                       parameters,
                                       input,
                                       output);
        return;
    }
    if (auto hopfield = layerAs<const HopfieldLayer>(layer)) {
        fillMatrixPreActivation(hopfield->get(), parameters, input, output);
        return;
    }

    throw std::runtime_error("Feedforward training requires pre-activation support.");
}

void activateInto(const Layer &layer,
                  const Pattern &values,
                  Pattern &output)
{
    const auto &activation = layer.getActivation();
    if (!activation) {
        throw std::runtime_error(
            "Activation function is not set for this layer.");
    }

    if (!output.hasSameShapeAs(values)) {
        throw std::runtime_error("Activation output shape mismatch.");
    }

    for (size_t i = 0; i < values.size(); ++i) {
        output[i] = (*activation)(values[i]);
    }
}

bool isFlattenLayer(const Layer &layer)
{
    return layerAs<const FlattenLayer>(layer).has_value();
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

void recordForwardPass(TrainingSession &session, const Pattern &input)
{
    auto &network = session.model();
    auto &activations = session.activations();
    auto &preActivations = session.preActivations();
    activations.at(0) = input;

    // Reuse session-owned tensors so each layer step avoids fresh temporaries.
    for (size_t layerIndex = 0; layerIndex < network.numLayers();
         ++layerIndex) {
        const auto &skill = network.getSkill(layerIndex);
        const auto &layer = skill.layer();
        const auto &current = activations.at(layerIndex);
        auto &nextActivation = activations.at(layerIndex + 1);
        if (isFlattenLayer(layer)) {
            fillFlattenOutput(current, nextActivation);
        } else {
            fillParameterizedPreActivation(skill,
                                           current,
                                           preActivations.at(layerIndex));
            activateInto(layer,
                         preActivations.at(layerIndex),
                         nextActivation);
        }
    }
}

void writeLossDerivative(const Pattern &output,
                         const Pattern &target,
                         Pattern &destination)
{
    if (!destination.hasSameShapeAs(output) || !output.hasSameShapeAs(target)) {
        throw std::runtime_error("Loss derivative shapes do not match.");
    }

    for (size_t i = 0; i < output.size(); ++i) {
        destination[i] = output[i] - target[i];
    }
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
BackpropagationPracticePlan::BackpropagationPracticePlan(
    BackpropagationPracticePlan &&) noexcept = default;
BackpropagationPracticePlan &BackpropagationPracticePlan::operator=(
    BackpropagationPracticePlan &&) noexcept = default;

void BackpropagationPracticePlan::practice(Model &network,
                                           const PracticeData &data,
                                           const PracticeOptions &options) const
{
    const auto &inputs = data.inputs;
    const auto &labels = data.labels;
    validateTrainingData(network, inputs, labels);

    TrainingSession session(network);
    initializeModelParameters(network);
    session.initializeForwardBuffers();

    std::cout << "Training feedforward Network..." << std::endl;
    for (size_t epoch = 0; epoch < options.epochs; ++epoch) {
        for (size_t sampleIndex = 0; sampleIndex < inputs.size();
             ++sampleIndex) {
            recordForwardPass(session, inputs.at(sampleIndex));
            writeLossDerivative(session.activations().back(),
                                labels.at(sampleIndex),
                                session.outputError());
            gradientEngine->computeLayerDeltas(
                network,
                session.activations(),
                session.preActivations(),
                session.outputError(),
                session.layerDeltas());

            optimizer->step(network,
                            session.activations(),
                            session.layerDeltas(),
                            options.learningRate);
        }
    }
}
