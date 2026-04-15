#include "base/Layer.h"

#include <random>
#include <stdexcept>

bool LayerConfig::isValid() const
{
    return learningRule && activation && inputSize > 0 && outputSize > 0;
}

Layer::Layer(const LayerConfig &newConfig)
    : config(newConfig)
{
    if (!config.isValid()) {
        throw std::invalid_argument("Invalid layer configuration");
    }
}

Layer::~Layer() = default;

Shape Layer::expectedWeightShape() const
{
    return {config.outputSize, config.inputSize};
}

Shape Layer::expectedBiasShape() const
{
    return {config.outputSize};
}

void Layer::initWeights(Scalar value)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Scalar> dis(-config.weightInitScale,
                                               config.weightInitScale);

    if (weights.empty()) {
        Pattern newWeights = Pattern::withShape(expectedWeightShape(), value);

        if (config.initWeights) {
            newWeights.generate([&dis, &gen]() { return dis(gen); });
        }

        setWeights(newWeights);
    }

    if (config.useBias && biases.empty()) {
        Pattern newBiases = Pattern::withShape(expectedBiasShape(), config.biasInit);

        if (config.initWeights) {
            newBiases.generate([&dis, &gen]() { return dis(gen); });
        }

        setBiases(newBiases);
    }
}

size_t Layer::getInputSize() const
{
    return config.inputSize;
}

size_t Layer::getOutputSize() const
{
    return config.outputSize;
}

const Pattern &Layer::getWeights() const
{
    return weights;
}

const Pattern &Layer::getBiases() const
{
    return biases;
}

void Layer::setWeights(const Pattern &newWeights)
{
    if (!newWeights.hasShape({config.outputSize, config.inputSize})) {
        throw std::runtime_error("Layer weights shape does not match layer configuration.");
    }

    weights = newWeights;
}

void Layer::setBiases(const Pattern &newBiases)
{
    if (newBiases.size() != config.outputSize) {
        throw std::runtime_error("Layer bias size does not match layer output size.");
    }

    biases = newBiases;
}

bool Layer::isInitialized() const
{
    return weights.hasShape({config.outputSize, config.inputSize})
           && (!config.useBias || biases.size() == config.outputSize);
}

void Layer::requireInitialized() const
{
    if (!isInitialized()) {
        throw std::runtime_error("Layer weights are not initialized.");
    }
}

void Layer::updateWeights(const Pattern &prev_activations,
                          const Pattern &layerDelta,
                          Scalar learningRate)
{
    requireInitialized();

    if (prev_activations.size() != config.inputSize) {
        throw std::runtime_error(
            "Previous activation size does not match layer input size.");
    }
    if (layerDelta.size() != config.outputSize) {
        throw std::runtime_error(
            "Layer delta size does not match layer output size.");
    }

    const Pattern weightGradients = layerDelta.outer(prev_activations);
    const auto updateValue = [this, learningRate](Scalar value, Scalar gradient) {
        return config.learningRule->updateWeight(value, gradient, learningRate);
    };

    weights = weights.zip(weightGradients, updateValue);

    if (config.useBias) {
        biases = biases.zipValues(layerDelta, updateValue);
    }
}

Pattern Layer::infer(const Pattern &input) const
{
    if (input.size() != config.inputSize) {
        throw std::runtime_error("Input size does not match layer input size.");
    }
    Pattern sums = weightedSum(input);
    return activate(sums);
}

Pattern Layer::weightedSum(const Pattern &input) const
{
    requireInitialized();

    if (input.empty()) {
        throw std::runtime_error("Input is empty");
    }

    Pattern sums = weights.matVec(input);
    return config.useBias ? sums + biases : sums;
}

Pattern Layer::activationDerivatives(const Pattern &values) const
{
    return values.map([this](Scalar value) {
        return (*config.activation).derivative(value);
    });
}

Pattern Layer::activate(const Pattern &values) const
{
    if (config.learningRule == nullptr) {
        throw std::runtime_error("Learning rule is not set for this layer.");
    }
    if (config.activation == nullptr) {
        throw std::runtime_error(
            "Activation function is not set for this layer.");
    }

    return values.map(
        [this](Scalar value) { return (*config.activation)(value); });
}

Pattern Layer::backwardPass(const Pattern &layerDelta,
                            const Pattern &preActivation)
{
    requireInitialized();
    return weights.transposedMatVec(layerDelta)
           * activationDerivatives(preActivation);
}

void Layer::naturalUpdateWeights(const Layer &l)
{
    requireInitialized();
    l.requireInitialized();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Scalar> dis(-0.01f, 0.01f);

    weights = l.weights.map([&dis, &gen](Scalar value) {
        return value + dis(gen);
    });

    if (config.useBias) {
        biases = l.biases.mapValues([&dis, &gen](Scalar value) {
            return value + dis(gen);
        });
    }
}
