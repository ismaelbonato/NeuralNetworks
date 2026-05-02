#include "base/ActivationFunction.h"
#include "base/LayerFactory.h"
#include "layers/DenseLayer.h"
#include "base/Model.h"
#include "training/ParameterInitializer.h"
#include "training/PerceptronRuleCoach.h"

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

TEST_CASE("perceptron inference uses coach-learned AND weights", "[perceptron]")
{
    auto skill = makePerceptronSkill();
    skill.setParameters({.weights = Pattern::matrix({{0.0F, 0.0F}}),
                         .biases = {0.0F}});

    Model network;
    network.addSkill(std::move(skill));
    const Batch inputs = {{0.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}};
    const Batch labels = {{0.0F}, {0.0F}, {0.0F}, {1.0F}};

    PerceptronRuleCoach coach;
    coach.learn(network, inputs, labels, 0.1F, 20);

    REQUIRE(network.infer({0.0F, 0.0F}).at(0) == 0.0F);
    REQUIRE(network.infer({0.0F, 1.0F}).at(0) == 0.0F);
    REQUIRE(network.infer({1.0F, 0.0F}).at(0) == 0.0F);
    REQUIRE(network.infer({1.0F, 1.0F}).at(0) == 1.0F);
}

TEST_CASE("perceptron coach learns AND gate", "[perceptron][coach]")
{
    auto skill = makePerceptronSkill();
    skill.setParameters({.weights = Pattern::matrix({{0.0F, 0.0F}}),
                         .biases = {0.0F}});

    Model network;
    network.addSkill(std::move(skill));
    PerceptronRuleCoach coach;
    const Batch inputs = {{0.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}};
    const Batch labels = {{0.0F}, {0.0F}, {0.0F}, {1.0F}};

    coach.learn(network, inputs, labels, 0.1F, 20);

    REQUIRE(network.infer({0.0F, 0.0F}).at(0) == 0.0F);
    REQUIRE(network.infer({0.0F, 1.0F}).at(0) == 0.0F);
    REQUIRE(network.infer({1.0F, 0.0F}).at(0) == 0.0F);
    REQUIRE(network.infer({1.0F, 1.0F}).at(0) == 1.0F);
}

TEST_CASE("perceptron coach initializes uninitialized layer",
          "[perceptron][coach]")
{
    DenseLayerRecipe config;
    config.name = "uninitialized perceptron";
    config.type = "DenseLayer";
    config.info = "coach initialization test layer";
    config.activation = std::make_shared<StepActivation<Scalar>>();
    config.inputSize = 2;
    config.outputSize = 1;

    Model network;
    network.addSkill(Skill(std::make_unique<DenseLayer>(config)));
    PerceptronRuleCoach coach;

    REQUIRE_NOTHROW(coach.learn(network,
                                  {{0.0F, 0.0F}},
                                  {{0.0F}},
                                  0.1F,
                                  1));
    REQUIRE_NOTHROW(network.getSkill(0).requireInitialized());
}

TEST_CASE("perceptron rejects multi-output layers", "[perceptron][errors]")
{
    Model network;
    network.addSkill(makePerceptronSkill(2));

    PerceptronRuleCoach coach;
    REQUIRE_THROWS_AS(coach.learn(network, {{1.0F, 1.0F}}, {{1.0F, 0.0F}}, 0.1F, 1),
                      std::runtime_error);
}

TEST_CASE("perceptron coach rejects invalid training data", "[perceptron][errors]")
{
    Model network;
    network.addSkill(makePerceptronSkill());

    PerceptronRuleCoach coach;
    REQUIRE_THROWS_AS(coach.learn(network, {}, {}, 0.1F, 1), std::runtime_error);
    REQUIRE_THROWS_AS(coach.learn(network, {{1.0F, 1.0F}}, {}, 0.1F, 1),
                      std::runtime_error);

    Model emptyNetwork;
    REQUIRE_THROWS_AS(coach.learn(emptyNetwork, {{1.0F, 1.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);
}
