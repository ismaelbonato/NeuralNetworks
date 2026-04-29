#include "training/Optimizer.h"

#include "base/Layer.h"
#include "base/Model.h"
#include "base/Skill.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/DenseLayer.h"
#include "layers/HopfieldLayer.h"

#include <functional>
#include <optional>
#include <typeinfo>

namespace {
template<typename LayerType>
std::optional<std::reference_wrapper<LayerType>> layerAs(Layer &layer)
{
    try {
        return std::ref(dynamic_cast<LayerType &>(layer));
    } catch (const std::bad_cast &) {
        return std::nullopt;
    }
}

void updateDenseLayer(Skill &skill,
                      DenseLayer &layer,
                      const Pattern &prevActivations,
                      const Pattern &layerDelta,
                      const Optimizer &optimizer,
                      Scalar learningRate)
{
    skill.requireInitialized();

    if (!prevActivations.hasShape(layer.getExpectedInputShape())) {
        throw std::runtime_error(
            "Previous activation shape does not match layer input shape.");
    }
    if (!layerDelta.hasShape(layer.getExpectedOutputShape())) {
        throw std::runtime_error(
            "Layer delta shape does not match layer output shape.");
    }

    const Pattern weightGradients = layerDelta.outer(prevActivations);
    const auto updateValue = [&optimizer, learningRate](Scalar value,
                                                        Scalar gradient) {
        return optimizer.update(value, gradient, learningRate);
    };

    LayerParameters parameters = skill.getParameters();
    parameters.weights = parameters.weights.zip(weightGradients, updateValue);

    if (!parameters.biases.empty()) {
        parameters.biases = parameters.biases.zipValues(layerDelta,
                                                        updateValue);
    }

    skill.setParameters(parameters);
}

void updateConvolutionalLayer(Skill &skill,
                              ConvolutionalLayer &layer,
                              const Pattern &prevActivations,
                              const Pattern &layerDelta,
                              const Optimizer &optimizer,
                              Scalar learningRate)
{
    skill.requireInitialized();

    if (!prevActivations.hasShape(layer.getExpectedInputShape())) {
        throw std::runtime_error("Previous activation shape does not match "
                                 "convolutional layer input shape.");
    }
    if (!layerDelta.hasShape(layer.getExpectedOutputShape())) {
        throw std::runtime_error("Layer delta shape does not match "
                                 "convolutional layer output shape.");
    }

    const auto &recipe = layer.getConvolutionalRecipe();
    Pattern weightGradients = Pattern::withShape(
        Shape(layer.getWeights().shape()),
        Scalar{0});
    Pattern biasGradients = Pattern::withShape(
        Shape(layer.getBiases().shape()),
        Scalar{0});

    for (size_t outputChannel = 0; outputChannel < recipe.outputChannels;
         ++outputChannel) {
        for (size_t outputIndex = 0; outputIndex < layerDelta.shape().at(1);
             ++outputIndex) {
            if (!biasGradients.empty()) {
                biasGradients.at(outputChannel) += layerDelta.at(
                    {outputChannel, outputIndex});
            }

            for (size_t inputChannel = 0; inputChannel < recipe.inputChannels;
                 ++inputChannel) {
                for (size_t kernelIndex = 0; kernelIndex < recipe.kernelSize;
                     ++kernelIndex) {
                    const size_t paddedInputIndex
                        = (outputIndex * recipe.stride) + kernelIndex;

                    if (paddedInputIndex < recipe.padding) {
                        continue;
                    }

                    const size_t inputIndex = paddedInputIndex
                                              - recipe.padding;
                    if (inputIndex >= recipe.inputLength) {
                        continue;
                    }

                    weightGradients.at(
                        {outputChannel, inputChannel, kernelIndex})
                        += layerDelta.at({outputChannel, outputIndex})
                           * prevActivations.at({inputChannel, inputIndex});
                }
            }
        }
    }

    const auto updateValue = [&optimizer, learningRate](Scalar value,
                                                        Scalar gradient) {
        return optimizer.update(value, gradient, learningRate);
    };

    LayerParameters parameters = skill.getParameters();
    parameters.weights = parameters.weights.zip(weightGradients, updateValue);

    if (!parameters.biases.empty()) {
        parameters.biases = parameters.biases.zipValues(biasGradients,
                                                        updateValue);
    }

    skill.setParameters(parameters);
}

void updateHopfieldLayer(Skill &skill,
                         HopfieldLayer &layer,
                         const Pattern &pattern,
                         const Optimizer &optimizer,
                         Scalar learningRate)
{
    skill.requireInitialized();

    const size_t patternSize = pattern.size();
    if (patternSize != layer.getInputSize()
        || patternSize != layer.getOutputSize()) {
        throw std::runtime_error(
            "Pattern size does not match Hopfield layer size.");
    }
    if (!pattern.hasShape(layer.getExpectedInputShape())) {
        throw std::runtime_error(
            "Pattern shape does not match Hopfield layer shape.");
    }

    Pattern weightGradients = pattern.outer(pattern);
    weightGradients.setDiagonal(Scalar{});

    LayerParameters parameters = skill.getParameters();
    parameters.weights = parameters.weights.zip(
        weightGradients,
        [&optimizer, learningRate](Scalar weight, Scalar gradient) {
            return optimizer.update(weight, gradient, learningRate);
        });
    parameters.weights.setDiagonal(Scalar{});

    skill.setParameters(parameters);
}
} // namespace

LearningRuleOptimizer::LearningRuleOptimizer(
    std::shared_ptr<LearningRule<Scalar>> newLearningRule)
    : learningRule(std::move(newLearningRule))
{
    if (!learningRule) {
        throw std::invalid_argument("Optimizer requires a learning rule.");
    }
}

Scalar LearningRuleOptimizer::update(Scalar value,
                                     Scalar gradient,
                                     Scalar learningRate) const
{
    return learningRule->updateWeight(value, gradient, learningRate);
}

void LearningRuleOptimizer::step(Model &network,
                                 const Batch &activations,
                                 const Batch &layerDeltas,
                                 Scalar learningRate) const
{
    for (size_t layerIndex = 0; layerIndex < network.numLayers(); ++layerIndex) {
        auto &skill = network.getSkill(layerIndex);
        auto &layer = skill.layer();
        if (auto dense = layerAs<DenseLayer>(layer)) {
            updateDenseLayer(skill,
                             dense->get(),
                             activations.at(layerIndex),
                             layerDeltas.at(layerIndex),
                             *this,
                             learningRate);
        } else if (auto convolutional = layerAs<ConvolutionalLayer>(layer)) {
            updateConvolutionalLayer(skill,
                                     convolutional->get(),
                                     activations.at(layerIndex),
                                     layerDeltas.at(layerIndex),
                                     *this,
                                     learningRate);
        } else if (auto hopfield = layerAs<HopfieldLayer>(layer)) {
            updateHopfieldLayer(skill,
                                hopfield->get(),
                                activations.at(layerIndex),
                                *this,
                                learningRate);
        }
    }
}
