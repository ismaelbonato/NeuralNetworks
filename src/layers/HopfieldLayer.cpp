#include "layers/HopfieldLayer.h"
#include "base/Layer.h"
#include <memory>
#include <stdexcept>

namespace nn {

Shape HopfieldLayerRecipe::getInputShape() const
{
    return stateShape;
}

Shape HopfieldLayerRecipe::getOutputShape() const
{
    return stateShape;
}

void HopfieldLayerRecipe::validateRecipe() const
{
    LayerRecipe::validateRecipe();

    if (!activation) {
        throw std::invalid_argument("Hopfield layer requires an activation.");
    }
}

HopfieldLayer::HopfieldLayer(const HopfieldLayerRecipe &newRecipe)
    : Layer(std::make_unique<HopfieldLayerRecipe>(newRecipe))
{}

HopfieldLayer::~HopfieldLayer() = default;

Shape HopfieldLayer::expectedWeightShape() const
{
    return {getOutputShape().elementCount(), getInputShape().elementCount()};
}

Shape HopfieldLayer::expectedBiasShape() const
{
    return {};
}

bool HopfieldLayer::hasBias() const
{
    return !expectedBiasShape().dimensions.empty();
}

bool HopfieldLayer::hasWeights() const
{
    return !expectedWeightShape().dimensions.empty();
}

Pattern HopfieldLayer::forward(const Pattern &input) const
{
    (void) input;
    throw std::runtime_error("Hopfield layer requires parameters.");
}

Pattern HopfieldLayer::forward(const Pattern &input,
                               const Parameters &parameters) const
{
    return recall(input, parameters);
}

Pattern HopfieldLayer::weightedInput(const Pattern &input,
                                     const Parameters &parameters) const
{
    if (input.empty()) {
        throw std::runtime_error("Input is empty");
    }
    Pattern sums = input.matVec(parameters.weights);
    return hasBias() ? sums + parameters.biases : sums;
}

Pattern HopfieldLayer::recall(const Pattern &input,
                              const Parameters &parameters) const
{
    if (!input.hasShape(getInputShape())) {
        throw std::runtime_error(
            "Input shape does not match Hopfield layer shape.");
    }

    Pattern state = input;
    Pattern prev_state;
    do {
        prev_state = state;
        auto sum = weightedInput(state, parameters);
        state = activate(sum);
    } while (state != prev_state);
    return state;
}

} // namespace nn
