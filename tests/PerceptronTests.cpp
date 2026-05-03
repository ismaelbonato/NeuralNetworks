#include "base/ActivationFunction.h"
#include "base/LayerFactory.h"
#include "layers/DenseLayer.h"
#include "base/Model.h"
#include "training/Coach.h"
#include "training/ParameterInitializer.h"
#include "training/PracticePlan.h"

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

Coach makePerceptronCoach()
{
    return Coach(std::make_unique<PerceptronRulePracticePlan>());
}

void practice(Coach &coach,
              Model &network,
              const Batch &inputs,
              const Batch &labels,
              Scalar learningRate,
              size_t epochs)
{
    coach.practice(network,
                   {.inputs = inputs, .labels = labels},
                   {.learningRate = learningRate, .epochs = epochs});
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

    Coach coach = makePerceptronCoach();
    practice(coach, network, inputs, labels, 0.1F, 20);

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
    Coach coach = makePerceptronCoach();
    const Batch inputs = {{0.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}};
    const Batch labels = {{0.0F}, {0.0F}, {0.0F}, {1.0F}};

    practice(coach, network, inputs, labels, 0.1F, 20);

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
    Coach coach = makePerceptronCoach();

    REQUIRE_NOTHROW(practice(coach,
                                  network,
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

    Coach coach = makePerceptronCoach();
    REQUIRE_THROWS_AS(practice(coach, network, {{1.0F, 1.0F}}, {{1.0F, 0.0F}}, 0.1F, 1),
                      std::runtime_error);
}

TEST_CASE("perceptron coach rejects invalid training data", "[perceptron][errors]")
{
    Model network;
    network.addSkill(makePerceptronSkill());

    Coach coach = makePerceptronCoach();
    REQUIRE_THROWS_AS(practice(coach, network, {}, {}, 0.1F, 1), std::runtime_error);
    REQUIRE_THROWS_AS(practice(coach, network, {{1.0F, 1.0F}}, {}, 0.1F, 1),
                      std::runtime_error);

    Model emptyNetwork;
    REQUIRE_THROWS_AS(practice(coach, emptyNetwork, {{1.0F, 1.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);
}
