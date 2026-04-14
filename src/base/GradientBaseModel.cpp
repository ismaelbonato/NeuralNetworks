#include "base/GradientBaseModel.h"

#include <stdexcept>

GradientBaseModel::GradientBaseModel() = default;

GradientBaseModel::GradientBaseModel(Layers &newlayers)
    : LayeredModel(newlayers)
{}

GradientBaseModel::GradientBaseModel(const std::initializer_list<std::shared_ptr<Layer>> &newlayers)
    : LayeredModel(newlayers)
{}

GradientBaseModel::~GradientBaseModel() = default;

void GradientBaseModel::forward(const Pattern &input)
{
    Pattern current = input;
    activate[0] = current;

    for (size_t i = 0; i < numLayers(); i++) {
        preActivations[i] = getLayer(i)->weightedSum(current);
        current = getLayer(i)->activate(preActivations[i]);
        activate[i + 1] = current;
    }
}

Pattern GradientBaseModel::computeOutputError(const Pattern &target)
{
    return lossDerivative(activate.back(), target);
}

void GradientBaseModel::backpropagation(const Pattern &outputError, Scalar rate)
{
    if (layers.empty()) {
        throw std::runtime_error("Cannot backpropagate without layers.");
    }

    Patterns layerDeltas(numLayers());

    layerDeltas.back() =
        outputError * layers.back()->activationDerivatives(preActivations.back());

    for (size_t l = numLayers() - 1; l > 0; --l) {
        layerDeltas[l - 1] =
            getLayer(l)->backwardPass(layerDeltas[l], preActivations[l - 1]);
    }

    for (size_t l = 0; l < numLayers(); ++l) {
        getLayer(l)->updateWeights(activate[l], layerDeltas[l], rate);
    }
}

Pattern GradientBaseModel::lossDerivative(const Pattern &output, const Pattern &target)
{
    return output - target;
}
