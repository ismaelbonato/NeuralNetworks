#include "base/ActivationFunction.h"
#include "base/LayerFactory.h"
#include "base/LearningRule.h"
#include "layers/DenseLayer.h"
#include "base/Model.h"
#include "training/NaturalSelectionTrainer.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
std::unique_ptr<DenseLayer> makePerceptronLayer()
{
    LayerConfig config{
        .learningRule = std::make_shared<PerceptronRule<Scalar>>(),
        .activation = std::make_shared<StepActivation<Scalar>>(),
        .inputSize = 2,
        .outputSize = 1,
        .name = "natural selection test perceptron",
        .type = "DenseLayer",
        .info = "deterministic test layer",
        .weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
        .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
    };

    return makeLayer<DenseLayer>(config);
}

std::unique_ptr<DenseLayer> makeMultiOutputLayer()
{
    LayerConfig config{
        .learningRule = std::make_shared<SGDRule<Scalar>>(),
        .activation = std::make_shared<SigmoidActivation<Scalar>>(),
        .inputSize = 2,
        .outputSize = 2,
        .name = "natural selection multi-output layer",
        .type = "DenseLayer",
        .info = "deterministic test layer",
        .weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
        .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
    };

    return makeLayer<DenseLayer>(config);
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

    auto layer = makePerceptronLayer();
    Model network;
    network.addLayer(std::move(layer));

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
    auto layer = makePerceptronLayer();
    Model network;
    network.addLayer(std::move(layer));
    NaturalSelectionTrainer trainer({.populationSize = 0});

    REQUIRE_THROWS_AS(trainer.learn(network, {{1.0F, 1.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);
}

TEST_CASE("natural selection trainer supports multi-output models",
          "[trainer][natural-selection]")
{
    auto layer = makeMultiOutputLayer();
    Model network;
    network.addLayer(std::move(layer));
    NaturalSelectionTrainer trainer({.populationSize = 2});

    REQUIRE_NOTHROW(trainer.learn(network,
                                  {{1.0F, 0.0F}},
                                  {{1.0F, 0.0F}},
                                  0.0F,
                                  1));
}
