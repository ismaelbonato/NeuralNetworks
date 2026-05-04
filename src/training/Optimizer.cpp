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
void ensurePatternShape(Pattern &buffer,
                        const Shape &shape,
                        Scalar fillValue = Scalar{})
{
    if (!buffer.hasShape(shape)) {
        buffer = Pattern::withShape(shape, fillValue);
        return;
    }

    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = fillValue;
    }
}

void ensurePatternStorage(Pattern &buffer, const Shape &shape)
{
    if (!buffer.hasShape(shape)) {
        buffer = Pattern::withShape(shape, Scalar{0});
    }
}

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
                      const LearningRuleOptimizer &optimizer,
                      Pattern &weightGradientScratch,
                      Parameters &parameterScratch,
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

    const Parameters &parameters = skill.getParameters();
    // Reuse optimizer scratch for gradients and updated parameters across steps.
    ensurePatternShape(weightGradientScratch,
                       static_cast<const Layer &>(layer).expectedWeightShape());
    ensurePatternStorage(parameterScratch.weights,
                         static_cast<const Layer &>(layer).expectedWeightShape());

    // gradient = outer(layerDelta, prevActivations)
    layerDelta.outerInto(prevActivations, weightGradientScratch);

    // Apply optimizer using flat contiguous storage.
    for (size_t i = 0; i < parameters.weights.size(); ++i) {
        parameterScratch.weights[i] = optimizer.update(parameters.weights[i],
                                                       weightGradientScratch[i],
                                                       learningRate);
    }

    if (!parameters.biases.empty()) {
        ensurePatternStorage(parameterScratch.biases,
                             layer.getExpectedOutputShape());
        for (size_t i = 0; i < parameters.biases.size(); ++i) {
            parameterScratch.biases.at(i)
                = optimizer.update(parameters.biases.at(i),
                                   layerDelta.at(i),
                                   learningRate);
        }
    } else {
        parameterScratch.biases = Pattern{};
    }

    skill.setParameters(parameterScratch);
}

void updateConvolutionalLayer(Skill &skill,
                              ConvolutionalLayer &layer,
                              const Pattern &prevActivations,
                              const Pattern &layerDelta,
                              const LearningRuleOptimizer &optimizer,
                              Pattern &weightGradientScratch,
                              Pattern &biasGradientScratch,
                              Parameters &parameterScratch,
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
    const Parameters &parameters = skill.getParameters();
    ensurePatternShape(weightGradientScratch, Shape(parameters.weights.shape()));
    ensurePatternShape(biasGradientScratch, Shape(parameters.biases.shape()));
    ensurePatternStorage(parameterScratch.weights,
                         Shape(parameters.weights.shape()));
    if (!parameters.biases.empty()) {
        ensurePatternStorage(parameterScratch.biases,
                             Shape(parameters.biases.shape()));
    }

    for (size_t outputChannel = 0; outputChannel < recipe.outputChannels;
         ++outputChannel) {
        for (size_t outputIndex = 0; outputIndex < layerDelta.shape().at(1);
             ++outputIndex) {
            if (!biasGradientScratch.empty()) {
                biasGradientScratch.at(outputChannel) += layerDelta.at(
                    {outputChannel, outputIndex});
            }

            for (size_t inputChannel = 0; inputChannel < recipe.inputChannels;
                 ++inputChannel) {
                for (size_t kernelIndex = 0; kernelIndex < recipe.kernelSize;
                     ++kernelIndex) {
                    const size_t paddedInputIndex = (outputIndex * recipe.stride)
                                                    + kernelIndex;

                    if (paddedInputIndex < recipe.padding) {
                        continue;
                    }

                    const size_t inputIndex = paddedInputIndex - recipe.padding;
                    if (inputIndex >= recipe.inputLength) {
                        continue;
                    }

                    weightGradientScratch.at(
                        {outputChannel, inputChannel, kernelIndex})
                        += layerDelta.at({outputChannel, outputIndex})
                           * prevActivations.at({inputChannel, inputIndex});
                }
            }
        }
    }

    for (size_t i = 0; i < parameters.weights.size(); ++i) {
        parameterScratch.weights[i] = optimizer.update(parameters.weights[i],
                                                       weightGradientScratch[i],
                                                       learningRate);
    }

    if (!parameters.biases.empty()) {
        for (size_t i = 0; i < parameters.biases.size(); ++i) {
            parameterScratch.biases[i] = optimizer.update(parameters.biases[i],
                                                          biasGradientScratch[i],
                                                          learningRate);
        }
    } else {
        parameterScratch.biases = Pattern{};
    }

    skill.setParameters(parameterScratch);
}

void updateHopfieldLayer(Skill &skill,
                         HopfieldLayer &layer,
                         const Pattern &pattern,
                         const LearningRuleOptimizer &optimizer,
                         Pattern &weightGradientScratch,
                         Parameters &parameterScratch,
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

    const Parameters &parameters = skill.getParameters();
    ensurePatternShape(weightGradientScratch,
                       static_cast<const Layer &>(layer).expectedWeightShape());
    ensurePatternStorage(parameterScratch.weights,
                         static_cast<const Layer &>(layer).expectedWeightShape());

    for (size_t row = 0; row < pattern.size(); ++row) {
        for (size_t col = 0; col < pattern.size(); ++col) {
            Scalar gradient = pattern.at(row) * pattern.at(col);
            if (row == col) {
                gradient = Scalar{};
            }
            weightGradientScratch.at({row, col}) = gradient;
            parameterScratch.weights.at({row, col})
                = optimizer.update(parameters.weights.at({row, col}),
                                   gradient,
                                   learningRate);
        }
    }
    parameterScratch.weights.setDiagonal(Scalar{});
    parameterScratch.biases = Pattern{};

    skill.setParameters(parameterScratch);
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
                             weightGradientScratch,
                             parameterScratch,
                             learningRate);
        } else if (auto convolutional = layerAs<ConvolutionalLayer>(layer)) {
            updateConvolutionalLayer(skill,
                                     convolutional->get(),
                                     activations.at(layerIndex),
                                     layerDeltas.at(layerIndex),
                                     *this,
                                     weightGradientScratch,
                                     biasGradientScratch,
                                     parameterScratch,
                                     learningRate);
        } else if (auto hopfield = layerAs<HopfieldLayer>(layer)) {
            updateHopfieldLayer(skill,
                                hopfield->get(),
                                activations.at(layerIndex),
                                *this,
                                weightGradientScratch,
                                parameterScratch,
                                learningRate);
        }
    }
}
