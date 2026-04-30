#include "base/LayerFactory.h"
#include "base/Model.h"
#include "base/Types.h"
#include "layers/DenseLayer.h"
#include "training/LayerParameterInitializer.h"
#include "training/LearningRule.h"
#include "training/Optimizer.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>

TEST_CASE("learning rules update weights according to their formulas", "[learning-rule]")
{
    SGDRule<Scalar> sgd;
    PerceptronRule<Scalar> perceptron;
    HebbianRule<Scalar> hebbian;

    REQUIRE(sgd.updateWeight(2.0F, 0.5F, 0.1F) == 1.95F);
    REQUIRE(perceptron.updateWeight(2.0F, 0.5F, 0.1F) == 2.05F);
    REQUIRE(hebbian.updateWeight(2.0F, 0.5F, 0.1F) == 2.5F);
}

TEST_CASE("learning-rule optimizer applies its learning rule", "[optimizer]")
{
    const LearningRuleOptimizer optimizer{
        std::make_shared<SGDRule<Scalar>>()};

    REQUIRE(optimizer.update(2.0F, 0.5F, 0.1F) == 1.95F);
}

TEST_CASE("optimizer step applies layer deltas to model parameters",
          "[optimizer][dense]")
{
    DenseLayerRecipe recipe;
    recipe.name = "optimizer dense";
    recipe.type = "DenseLayer";
    recipe.activation = std::make_shared<IdentityActivation<Scalar>>();
    recipe.inputSize = 1;
    recipe.outputSize = 1;

    Model network;
    network.addLayer(makeInitializedLayer<DenseLayer>(
        recipe,
        {.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
         .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>()}));

    Batch activations(2);
    activations.at(0) = Pattern{3.0F};
    activations.at(1) = Pattern{0.0F};

    Batch layerDeltas(1);
    layerDeltas.at(0) = Pattern{2.0F};

    const LearningRuleOptimizer optimizer{
        std::make_shared<SGDRule<Scalar>>()};
    optimizer.step(network, activations, layerDeltas, 0.5F);

    const LayerParameters parameters = network.getSkill(0).getParameters();
    REQUIRE(parameters.weights.at({0, 0}) == -3.0F);
    REQUIRE(parameters.biases.at(0) == -1.0F);
}
