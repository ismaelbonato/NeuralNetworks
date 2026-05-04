#include "base/ActivationFunction.h"
#include "base/LayerFactory.h"
#include "layers/DenseLayer.h"
#include "base/Model.h"
#include "training/ParameterInitializer.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace
{
Skill makePerceptronSkill(const size_t outputSize = 1)
{
    DenseLayerRecipe config;
    config.name = "test perceptron";
    config.type = "DenseLayer";
    config.info = "deterministic test layer";
    config.activation = std::make_shared<StepActivation<Scalar>>();
    config.inputSize = 2;
    config.outputSize = outputSize;

    return makeTrainableSkill<DenseLayer>(
        config,
        {.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
         .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>()})
        .intoSkill();
}

}

TEST_CASE("perceptron inference uses static AND fixture weights",
          "[perceptron][runtime]")
{
    auto skill = makePerceptronSkill();
    skill.setParameters({.weights = Pattern::matrix({{0.2F, 0.1F}}),
                         .biases = {-0.3F}});

    Model network;
    network.addSkill(std::move(skill));

    REQUIRE(network.infer({0.0F, 0.0F}).at(0) == 0.0F);
    REQUIRE(network.infer({0.0F, 1.0F}).at(0) == 0.0F);
    REQUIRE(network.infer({1.0F, 0.0F}).at(0) == 0.0F);
    REQUIRE(network.infer({1.0F, 1.0F}).at(0) == 1.0F);
}
