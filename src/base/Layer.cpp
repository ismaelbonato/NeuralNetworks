#include "base/Layer.h"

#include <memory>
#include <stdexcept>

namespace nn {

void LayerRecipe::validateRecipe() const
{
    if (!getInputShape().isValid() || !getOutputShape().isValid()) {
        throw std::invalid_argument("Layer recipe requires valid shapes.");
    }
}

Layer::Layer(std::unique_ptr<LayerRecipe> newRecipe)
    : recipe(std::move(newRecipe))
{
    recipe->validateRecipe();
    ownedParameters = Parameters{};
}

Layer::~Layer() = default;

Shape Layer::getInputShape() const
{
    return recipe->getInputShape();
}

Shape Layer::getOutputShape() const
{
    return recipe->getOutputShape();
}

const std::shared_ptr<ActivationFunction<Scalar>> &Layer::getActivation() const
{
    return recipe->activation;
}

const std::string &Layer::getName() const
{
    return recipe->name;
}

const std::string &Layer::getType() const
{
    return recipe->type;
}

const std::string &Layer::getInfo() const
{
    return recipe->info;
}

bool Layer::usesParameters() const
{
    return !expectedWeightShape().empty() || !expectedBiasShape().empty();
}

Shape Layer::expectedWeightShape() const
{
    return {};
}

Shape Layer::expectedBiasShape() const
{
    return {};
}

void Layer::requireValidParameters(const Parameters &parameters) const
{
    const Shape weightShape = expectedWeightShape();
    const Shape biasShape = expectedBiasShape();

    const bool weightsValid = weightShape.empty()
                                  ? parameters.weights.empty()
                                  : parameters.weights.hasShape(weightShape);
    const bool biasesValid = biasShape.empty()
                                 ? parameters.biases.empty()
                                 : parameters.biases.hasShape(biasShape);

    if (!weightsValid || !biasesValid) {
        throw std::runtime_error(
            "Layer parameters do not match expected shapes.");
    }
}

std::optional<Parameters> Layer::parameters() const
{
    if (!usesParameters()) {
        return std::nullopt;
    }

    return ownedParameters;
}

const Parameters &Layer::getParameters() const
{
    if (usesParameters() && ownedParameters) {
        return *ownedParameters;
    }

    throw std::runtime_error("Layer does not expose parameters.");
}

void Layer::setParameters(const Parameters &parameters)
{
    if (!usesParameters()) {
        throw std::runtime_error("Layer does not accept parameters.");
    }

    requireValidParameters(parameters);
    ownedParameters = parameters;
}

void Layer::requireParameters() const
{
    if (!usesParameters()) {
        return;
    }

    requireValidParameters(getParameters());
}

void Layer::requireInputShape(const Pattern &input) const
{
    if (!input.hasShape(getInputShape())) {
        throw std::runtime_error(
            "Input shape does not match layer input shape.");
    }
}

Pattern Layer::infer(const Pattern &input) const
{
    requireInputShape(input);
    if (usesParameters()) {
        return forward(input, getParameters());
    }
    return forward(input);
}

Pattern Layer::forward(const Pattern &input) const
{
    (void) input;
    throw std::runtime_error("Layer requires parameters.");
}

Pattern Layer::forward(const Pattern &input, const Parameters &parameters) const
{
    return activate(weightedInput(input, parameters));
}

Pattern Layer::weightedInput(const Pattern &input,
                             const Parameters &parameters) const
{
    (void) input;
    (void) parameters;
    throw std::runtime_error("Layer does not implement weighted input.");
}

Pattern Layer::activate(const Pattern &values) const
{
    if (recipe->activation == nullptr) {
        throw std::runtime_error(
            "Activation function is not set for this layer.");
    }

    return values.map(
        [this](Scalar value) { return (*recipe->activation)(value); });
}

} // namespace nn
