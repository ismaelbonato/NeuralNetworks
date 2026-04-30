#include "base/ActivationFunction.h"
#include "base/LayerFactory.h"
#include "layers/DenseLayer.h"
#include "base/Model.h"
#include "training/LayerParameterInitializer.h"
#include "training/NaturalSelectionTrainer.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
Skill makePerceptronSkill()
{
    DenseLayerRecipe config;
    config.name = "natural selection test perceptron";
    config.type = "DenseLayer";
    config.info = "deterministic test layer";
    config.activation = std::make_shared<StepActivation<Scalar>>();
    config.inputSize = 2;
    config.outputSize = 1;

    return makeTrainableSkill<DenseLayer>(
        config,
        {.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
         .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>()})
        .intoSkill();
}

Skill makeMultiOutputSkill()
{
    DenseLayerRecipe config;
    config.name = "natural selection multi-output layer";
    config.type = "DenseLayer";
    config.info = "deterministic test layer";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputSize = 2;
    config.outputSize = 2;

    return makeTrainableSkill<DenseLayer>(
        config,
        {.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
         .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>()})
        .intoSkill();
}
}

TEST_CASE("natural selection trainer selects candidate with lowest squared error",
          "[perceptron][trainer][natural-selection]")
{
    NaturalSelectionTrainer trainer;
    const std::vector<Batch> candidatePredictions = {
        {{0.0F}, {0.0F}, {0.0F}, {1.0F}},
        {{1.0F}, {1.0F}, {1.0F}, {1.0F}},
        {{0.0F}, {1.0F}, {1.0F}, {0.0F}},
        {{0.0F}, {0.0F}, {1.0F}, {1.0F}},
    };
    const Batch labels = {{0.0F}, {0.0F}, {0.0F}, {1.0F}};

    REQUIRE(trainer.findBestCandidate(candidatePredictions, labels) == 0);
}

TEST_CASE("natural selection trainer scores full output patterns",
          "[trainer][natural-selection]")
{
    NaturalSelectionTrainer trainer;
    const std::vector<Batch> candidatePredictions = {
        {{0.0F, 10.0F}},
        {{1.0F, 1.0F}},
    };
    const Batch labels = {{0.0F, 1.0F}};

    REQUIRE(trainer.findBestCandidate(candidatePredictions, labels) == 1);
}

TEST_CASE("natural selection trainer rejects invalid training data",
          "[perceptron][trainer][natural-selection][errors]")
{
    NaturalSelectionTrainer trainer;
    Model emptyNetwork;

    REQUIRE_THROWS_AS(trainer.learn(emptyNetwork, {{1.0F, 1.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);

    Model network;
    network.addSkill(makePerceptronSkill());

    REQUIRE_THROWS_AS(trainer.learn(network, {}, {}, 0.1F, 1), std::runtime_error);
    REQUIRE_THROWS_AS(trainer.learn(network, {{1.0F, 1.0F}}, {}, 0.1F, 1),
                      std::runtime_error);
    REQUIRE_THROWS_AS(trainer.learn(network, {{1.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);
    REQUIRE_THROWS_AS(trainer.learn(network, {{1.0F, 1.0F}}, {{1.0F, 0.0F}}, 0.1F, 1),
                      std::runtime_error);
}

TEST_CASE("natural selection trainer rejects invalid configuration",
          "[trainer][natural-selection][errors]")
{
    Model network;
    network.addSkill(makePerceptronSkill());
    NaturalSelectionTrainer trainer({.populationSize = 0});

    REQUIRE_THROWS_AS(trainer.learn(network, {{1.0F, 1.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);
}

TEST_CASE("natural selection trainer supports multi-output models",
          "[trainer][natural-selection]")
{
    Model network;
    network.addSkill(makeMultiOutputSkill());
    NaturalSelectionTrainer trainer({.populationSize = 2});

    REQUIRE_NOTHROW(trainer.learn(network,
                                  {{1.0F, 0.0F}},
                                  {{1.0F, 0.0F}},
                                  0.0F,
                                  1));
}
