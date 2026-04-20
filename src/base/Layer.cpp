#include "base/Layer.h"

#include <random>
#include <stdexcept>

namespace
{
bool isSizeCompatibleWithShape(const size_t size, const Shape &shape)
{
    return size == 0 || shape.empty() || (shape.isValid() && shape.elementCount() == size);
}

}

bool DenseLayerConfig::isValid() const
{
    return learningRule && activation
           && (inputSize > 0 || expectedInputShape.isValid())
           && (outputSize > 0 || expectedOutputShape.isValid())
           && isSizeCompatibleWithShape(inputSize, expectedInputShape)
           && isSizeCompatibleWithShape(outputSize, expectedOutputShape);
}

bool HopfieldLayerConfig::isValid() const
{
    return learningRule && activation
           && (size > 0 || expectedShape.isValid())
           && isSizeCompatibleWithShape(size, expectedShape);
}

bool FlattenLayerConfig::isValid() const
{
    return expectedInputShape.isValid();
}

Shape FlattenLayerConfig::expectedOutputShape() const
{
    return {expectedInputShape.elementCount()};
}

Layer::Layer(const LayerConfig &newConfig,
             const Shape &newExpectedInput,
             const Shape &newExpectedOutput)
    : config(newConfig),
      expectedInput(newExpectedInput),
      expectedOutput(newExpectedOutput)
{
    if (!expectedInput.isValid() || !expectedOutput.isValid()) {
        throw std::invalid_argument("Invalid layer configuration");
    }
}

Layer::~Layer() = default;

size_t Layer::getInputSize() const
{
    return expectedInput.elementCount();
}

size_t Layer::getOutputSize() const
{
    return expectedOutput.elementCount();
}

const Shape &Layer::getExpectedInputShape() const
{
    return expectedInput;
}

const Shape &Layer::getExpectedOutputShape() const
{
    return expectedOutput;
}

const Shape &Layer::getInputShape() const
{
    return getExpectedInputShape();
}

const Shape &Layer::getOutputShape() const
{
    return getExpectedOutputShape();
}

bool Layer::isTrainable() const
{
    return false;
}

void Layer::requireInputShape(const Pattern &input) const
{
    if (!input.hasShape(expectedInput)) {
        throw std::runtime_error("Input shape does not match layer input shape.");
    }
}

TrainableLayer::TrainableLayer(const TrainableLayerConfig &newConfig,
                               const Shape &newExpectedInput,
                               const Shape &newExpectedOutput)
    : Layer(newConfig, newExpectedInput, newExpectedOutput),
      trainableConfig(newConfig)
{
    if (!trainableConfig.learningRule || !trainableConfig.activation) {
        throw std::invalid_argument("Invalid trainable layer configuration");
    }
}

TrainableLayer::~TrainableLayer() = default;

bool TrainableLayer::isTrainable() const
{
    return true;
}

bool TrainableLayer::hasBias() const
{
    return !expectedBiasShape().dimensions.empty();
}

bool TrainableLayer::hasWeights() const
{
    return !expectedWeightShape().dimensions.empty();
}

Pattern TrainableLayer::initializeParameter(const Shape &shape,
                                            const std::shared_ptr<Initializer<Scalar>> &initializer,
                                            Scalar fallbackValue)
{
    Pattern parameter = Pattern::withShape(shape, fallbackValue);

    if (initializer) {
        initializer->fill(parameter);
    }

    return parameter;
}

void TrainableLayer::initializeParameters(Scalar value)
{
    if (hasWeights() && weights.empty()) {
        setWeights(initializeParameter(expectedWeightShape(),
                                       trainableConfig.weightInitializer,
                                       value));
    }

    if (hasBias() && biases.empty()) {
        setBiases(initializeParameter(expectedBiasShape(),
                                      trainableConfig.biasInitializer,
                                      Scalar{}));
    }
}

const Pattern &TrainableLayer::getWeights() const
{
    return weights;
}

const Pattern &TrainableLayer::getBiases() const
{
    return biases;
}

LayerParameters TrainableLayer::getParameters() const
{
    return {
        .weights = weights,
        .biases = biases,
    };
}

void TrainableLayer::setParameters(const LayerParameters &parameters)
{
    setWeights(parameters.weights);
    setBiases(parameters.biases);
}

void TrainableLayer::setWeights(const Pattern &newWeights)
{
    if (!hasWeights()) {
        if (!newWeights.empty()) {
            throw std::runtime_error("Layer does not use weights.");
        }

        weights = newWeights;
        return;
    }

    if (!newWeights.hasShape(expectedWeightShape())) {
        throw std::runtime_error("Layer weights shape does not match layer configuration.");
    }

    weights = newWeights;
}

void TrainableLayer::setBiases(const Pattern &newBiases)
{
    if (!hasBias()) {
        if (!newBiases.empty()) {
            throw std::runtime_error("Layer does not use bias.");
        }

        biases = newBiases;
        return;
    }

    if (!newBiases.hasShape(expectedBiasShape())) {
        throw std::runtime_error("Layer bias size does not match layer output size.");
    }

    biases = newBiases;
}

bool TrainableLayer::isInitialized() const
{
    return (!hasWeights() || weights.hasShape(expectedWeightShape()))
           && (!hasBias() || biases.hasShape(expectedBiasShape()));
}

void TrainableLayer::requireInitialized() const
{
    if (!isInitialized()) {
        throw std::runtime_error("Layer weights are not initialized.");
    }
}

void TrainableLayer::updateWeights(const Pattern &prev_activations,
                                   const Pattern &layerDelta,
                                   Scalar learningRate)
{
    requireInitialized();

    if (!prev_activations.hasShape(expectedInput)) {
        throw std::runtime_error(
            "Previous activation shape does not match layer input shape.");
    }
    if (!layerDelta.hasShape(expectedOutput)) {
        throw std::runtime_error(
            "Layer delta shape does not match layer output shape.");
    }

    const Pattern weightGradients = layerDelta.outer(prev_activations);
    const auto updateValue = [this, learningRate](Scalar value, Scalar gradient) {
        return trainableConfig.learningRule->updateWeight(value, gradient, learningRate);
    };

    weights = weights.zip(weightGradients, updateValue);

    if (hasBias()) {
        biases = biases.zipValues(layerDelta, updateValue);
    }
}

Pattern TrainableLayer::infer(const Pattern &input) const
{
    requireInputShape(input);
    Pattern sums = weightedSum(input);
    return activate(sums);
}

Pattern TrainableLayer::weightedSum(const Pattern &input) const
{
    requireInitialized();

    if (input.empty()) {
        throw std::runtime_error("Input is empty");
    }

    Pattern sums = weights.matVec(input);
    return hasBias() ? sums + biases : sums;
}

Pattern TrainableLayer::activationDerivatives(const Pattern &values) const
{
    return values.map([this](Scalar value) {
        return (*trainableConfig.activation).derivative(value);
    });
}

Pattern TrainableLayer::activate(const Pattern &values) const
{
    if (trainableConfig.learningRule == nullptr) {
        throw std::runtime_error("Learning rule is not set for this layer.");
    }
    if (trainableConfig.activation == nullptr) {
        throw std::runtime_error(
            "Activation function is not set for this layer.");
    }

    return values.map(
        [this](Scalar value) { return (*trainableConfig.activation)(value); });
}

Pattern TrainableLayer::backwardPass(const Pattern &layerDelta,
                                     const Pattern &preActivation) const
{
    requireInitialized();
    return weights.transposedMatVec(layerDelta)
           * activationDerivatives(preActivation);
}

LayerParameters TrainableLayer::naturalUpdatedParameters(const LayerParameters &parameters,
                                                        Scalar mutationStrength) const
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Scalar> dis(-mutationStrength, mutationStrength);

    LayerParameters updated{
        .weights = parameters.weights.map([&dis, &gen](Scalar value) {
            return value + dis(gen);
        }),
        .biases = parameters.biases,
    };

    if (hasBias()) {
        updated.biases = parameters.biases.mapValues([&dis, &gen](Scalar value) {
            return value + dis(gen);
        });
    }

    return updated;
}
