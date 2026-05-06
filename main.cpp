#include "base/ActivationFunction.h"
#include "base/Model.h"
#include "layers/DenseLayer.h"

#include <iostream>

using namespace nn;

namespace {
std::unique_ptr<DenseLayer> makeDenseLayer(const size_t inputSize,
                                           const size_t outputSize,
                                           const Parameters &parameters)
{
    DenseLayerRecipe recipe;
    recipe.name = "runtime dense layer";
    recipe.type = "DenseLayer";
    recipe.info = "runtime XOR fixture layer";
    recipe.activation = std::make_shared<SigmoidActivation<Scalar>>();
    recipe.inputSize = inputSize;
    recipe.outputSize = outputSize;

    auto layer = std::make_unique<DenseLayer>(recipe);
    layer->setParameters(parameters);
    return layer;
}
} // namespace

int main()
{
    Model network;
    network.addLayer(
        makeDenseLayer(2,
                       2,
                       {.weights = Pattern::matrix(
                            {{8.051888F, 8.051895F}, {-8.016418F, -8.016412F}}),
                        .biases = {-3.967814F, 12.036060F}}));
    network.addLayer(
        makeDenseLayer(2,
                       1,
                       {.weights = Pattern::matrix({{8.243962F, 8.242302F}}),
                        .biases = {-12.212015F}}));

    std::cout << "Runtime XOR fixture" << std::endl;
    for (const Pattern &input : {Pattern{0.0F, 0.0F},
                                 Pattern{0.0F, 1.0F},
                                 Pattern{1.0F, 0.0F},
                                 Pattern{1.0F, 1.0F}}) {
        std::cout << input << " -> " << network.infer(input) << std::endl;
    }

    return 0;
}
