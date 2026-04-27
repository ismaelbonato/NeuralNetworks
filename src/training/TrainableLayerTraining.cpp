#include "base/Layer.h"

#include <random>
#include <stdexcept>

Pattern TrainableLayer::activationDerivatives(const Pattern &values) const
{
    return values.map([this](Scalar value) {
        return (*trainableConfig.activation).derivative(value);
    });
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
    const auto updateValue = [this, learningRate](Scalar value,
                                                  Scalar gradient) {
        return trainableConfig.learningRule->updateWeight(value,
                                                          gradient,
                                                          learningRate);
    };

    weights = weights.zip(weightGradients, updateValue);

    if (hasBias()) {
        biases = biases.zipValues(layerDelta, updateValue);
    }
}

Pattern TrainableLayer::backwardPass(const Pattern &layerDelta,
                                     const Pattern &layerInput) const
{
    requireInitialized();
    if (!layerDelta.hasShape(expectedOutput)) {
        throw std::runtime_error(
            "Layer delta shape does not match layer output shape.");
    }
    if (!layerInput.hasShape(expectedInput)) {
        throw std::runtime_error(
            "Layer input shape does not match layer input shape.");
    }
    return weights.transposedMatVec(layerDelta);
}

LayerParameters TrainableLayer::naturalUpdatedParameters(
    const LayerParameters &parameters, Scalar mutationStrength) const
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Scalar> dis(-mutationStrength,
                                               mutationStrength);

    LayerParameters updated{
        .weights = parameters.weights.map(
            [&dis, &gen](Scalar value) { return value + dis(gen); }),
        .biases = parameters.biases,
    };

    if (hasBias()) {
        updated.biases = parameters.biases.mapValues(
            [&dis, &gen](Scalar value) { return value + dis(gen); });
    }

    return updated;
}
