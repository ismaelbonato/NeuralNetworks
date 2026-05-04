#include "base/ActivationFunction.h"
#include "base/LayerFactory.h"
#include "layers/DenseLayer.h"
#include "base/Model.h"
#include "training/Coach.h"
#include "training/ParameterInitializer.h"
#include "training/TrainingSession.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace
{
constexpr Scalar tolerance = 0.0001F;

std::unique_ptr<DenseLayer> makeDenseLayer(const size_t inputSize,
                                           const size_t outputSize,
                                           const bool randomInitialize = false)
{
    auto activation = std::make_shared<SigmoidActivation<Scalar>>();
    DenseLayerRecipe config;
    config.name = "test dense layer";
    config.type = "DenseLayer";
    config.info = "deterministic test layer";
    config.activation = activation;
    config.inputSize = inputSize;
    config.outputSize = outputSize;

    auto layer = makeLayer<DenseLayer>(config);
    (void)randomInitialize;
    return layer;
}

Skill makeDenseSkill(const size_t inputSize,
                     const size_t outputSize,
                     const bool randomInitialize = false)
{
    auto activation = std::make_shared<SigmoidActivation<Scalar>>();
    DenseLayerRecipe config;
    config.name = "test dense skill";
    config.type = "DenseLayer";
    config.info = "deterministic test skill";
    config.activation = activation;
    config.inputSize = inputSize;
    config.outputSize = outputSize;

    return (randomInitialize
                ? makeTrainableSkill<DenseLayer>(config)
                : makeTrainableSkill<DenseLayer>(
                      config,
                      {.weightInitializer
                       = std::make_shared<ZeroInitializer<Scalar>>(),
                       .biasInitializer
                       = std::make_shared<ZeroInitializer<Scalar>>()}))
        .intoSkill();
}

Skill makeDenseSkill(const size_t inputSize,
                     const size_t outputSize,
                     const Parameters &parameters)
{
    Skill skill = makeDenseSkill(inputSize, outputSize);
    skill.setParameters(parameters);
    return skill;
}

void requireClose(const Scalar actual, const Scalar expected)
{
    REQUIRE(std::fabs(actual - expected) < tolerance);
}

Parameters denseParameters(Model &network, const size_t index)
{
    return network.getSkill(index).getParameters();
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

TEST_CASE("dense layer computes deterministic pre-activations and activations",
          "[feedforward][dense]")
{
    auto layer = makeDenseLayer(2, 2);
    const Parameters parameters{
        .weights = Pattern::matrix({{1.0F, -1.0F}, {0.5F, 0.5F}}),
        .biases = {0.0F, -0.5F},
    };

    const Pattern output = layer->infer({2.0F, 1.0F}, parameters);

    requireClose(output.at(0), 0.7310586F);
    requireClose(output.at(1), 0.7310586F);
}

TEST_CASE("feedforward inference composes dense layers", "[feedforward]")
{
    Model network;
    network.addSkill(makeDenseSkill(
        2,
        1,
        {.weights = Pattern::matrix({{1.0F, 1.0F}}), .biases = {0.0F}}));
    network.addSkill(makeDenseSkill(
        1,
        1,
        {.weights = Pattern::matrix({{2.0F}}), .biases = {-1.0F}}));

    const Pattern prediction = network.infer({1.0F, 1.0F});

    requireClose(prediction.at(0), 0.6816998F);
}

TEST_CASE("feedforward inference uses static OR fixture weights",
          "[feedforward][dense][runtime]")
{
    Model network;
    network.addSkill(makeDenseSkill(
        2,
        1,
        {.weights = Pattern::matrix({{10.0F, 10.0F}}), .biases = {-5.0F}}));

    REQUIRE(network.infer({0.0F, 0.0F}).at(0) < 0.1F);
    REQUIRE(network.infer({0.0F, 1.0F}).at(0) > 0.9F);
    REQUIRE(network.infer({1.0F, 0.0F}).at(0) > 0.9F);
    REQUIRE(network.infer({1.0F, 1.0F}).at(0) > 0.9F);
}

TEST_CASE("feedforward inference uses static AND fixture weights",
          "[feedforward][dense][runtime]")
{
    Model network;
    network.addSkill(makeDenseSkill(
        2,
        1,
        {.weights = Pattern::matrix({{10.0F, 10.0F}}), .biases = {-15.0F}}));

    REQUIRE(network.infer({0.0F, 0.0F}).at(0) < 0.1F);
    REQUIRE(network.infer({0.0F, 1.0F}).at(0) < 0.1F);
    REQUIRE(network.infer({1.0F, 0.0F}).at(0) < 0.1F);
    REQUIRE(network.infer({1.0F, 1.0F}).at(0) > 0.9F);
}

TEST_CASE("feedforward coach updates single layer through SGD",
          "[feedforward][learning]")
{
    Model network;
    network.addSkill(makeDenseSkill(1, 1));
    Coach coach;

    practice(coach, network, {{1.0F}}, {{1.0F}}, 1.0F, 1);

    const Parameters parameters = denseParameters(network, 0);
    requireClose(parameters.weights.at({0, 0}), 0.125F);
    requireClose(parameters.biases.at(0), 0.125F);
}

TEST_CASE("feedforward inference rejects missing layers and invalid input sizes",
          "[feedforward][errors]")
{
    Model emptyNetwork;

    REQUIRE_THROWS_AS(emptyNetwork.infer({1.0F}), std::runtime_error);

    Model network;
    network.addSkill(makeDenseSkill(2, 1));

    REQUIRE_THROWS_AS(network.infer({1.0F}), std::runtime_error);
}

TEST_CASE("feedforward coach can train the same model more than once",
          "[feedforward][learning]")
{
    Model network;
    network.addSkill(makeDenseSkill(1, 1));
    Coach coach;

    practice(coach, network, {{1.0F}}, {{1.0F}}, 1.0F, 1);
    practice(coach, network, {{1.0F}}, {{1.0F}}, 1.0F, 1);

    REQUIRE(denseParameters(network, 0).weights.at({0, 0}) != 0.0F);
}

TEST_CASE("feedforward coach direct API updates weights and biases",
          "[feedforward][coach]")
{
    Model network;
    network.addSkill(makeDenseSkill(1, 1));
    Coach coach;

    practice(coach, network, {{1.0F}}, {{1.0F}}, 1.0F, 1);

    const Parameters parameters = denseParameters(network, 0);
    requireClose(parameters.weights.at({0, 0}), 0.125F);
    requireClose(parameters.biases.at(0), 0.125F);
}

TEST_CASE("training session initializes forward buffers from model skills",
          "[training][session]")
{
    Model network;
    network.addSkill(makeDenseSkill(1, 1));

    TrainingSession session(network);
    session.initializeForwardBuffers();

    REQUIRE(session.activations().size() == 2);
    REQUIRE(session.preActivations().size() == 1);
    REQUIRE(session.layerDeltas().size() == 1);
    REQUIRE(session.activations().at(0).hasShape({1}));
    REQUIRE(session.activations().at(1).hasShape({1}));
    REQUIRE(session.preActivations().at(0).hasShape({1}));
    REQUIRE(session.layerDeltas().at(0).hasShape({1}));
    REQUIRE(session.outputError().hasShape({1}));
}

TEST_CASE("generic coach preserves feedforward training behavior",
          "[feedforward][coach]")
{
    Model network;
    network.addSkill(makeDenseSkill(1, 1));
    Coach coach;

    practice(coach, network, {{1.0F}}, {{1.0F}}, 1.0F, 1);

    const Parameters parameters = denseParameters(network, 0);
    requireClose(parameters.weights.at({0, 0}), 0.125F);
    requireClose(parameters.biases.at(0), 0.125F);
}

TEST_CASE("feedforward coach updates hidden and output layers",
          "[feedforward][learning]")
{
    auto hidden = makeDenseSkill(
        2,
        2,
        {.weights = Pattern::matrix({{0.1F, -0.2F}, {0.3F, 0.4F}}),
         .biases = {0.0F, 0.0F}});
    auto output = makeDenseSkill(
        2,
        1,
        {.weights = Pattern::matrix({{0.5F, -0.3F}}),
         .biases = {0.0F}});

    const Scalar hiddenWeightBefore = hidden.getParameters().weights.at({0, 0});
    const Scalar outputWeightBefore = output.getParameters().weights.at({0, 0});

    Model network;
    network.addSkill(std::move(hidden));
    network.addSkill(std::move(output));
    Coach coach;
    practice(coach, network, {{1.0F, 0.0F}}, {{1.0F}}, 0.5F, 1);

    REQUIRE(denseParameters(network, 0).weights.at({0, 0}) != hiddenWeightBefore);
    REQUIRE(denseParameters(network, 1).weights.at({0, 0}) != outputWeightBefore);
}

TEST_CASE("feedforward coach rejects invalid training data", "[feedforward][errors]")
{
    Model network;
    network.addSkill(makeDenseSkill(1, 1));
    Coach coach;

    REQUIRE_THROWS_AS(practice(coach, network, {}, {}, 0.1F, 1), std::runtime_error);
    REQUIRE_THROWS_AS(practice(coach, network, {{1.0F}}, {}, 0.1F, 1), std::runtime_error);

    Model emptyNetwork;
    REQUIRE_THROWS_AS(practice(coach, emptyNetwork, {{1.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);
}

TEST_CASE("feedforward coach rejects wrong input and label shapes", "[feedforward][errors]")
{
    Model network;
    network.addSkill(makeDenseSkill(2, 2));
    Coach coach;

    REQUIRE_THROWS_AS(practice(coach, network, {{1.0F}}, {{1.0F, 0.0F}}, 0.1F, 1),
                      std::runtime_error);
    REQUIRE_THROWS_AS(practice(coach, network, {{1.0F, 0.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);
    REQUIRE_THROWS_AS(practice(coach, network, {{1.0F, 0.0F}, {1.0F}},
                                    {{1.0F, 0.0F}, {0.0F, 1.0F}},
                                    0.1F,
                                    1),
                      std::runtime_error);
    REQUIRE_THROWS_AS(practice(coach, network, {{1.0F, 0.0F}, {0.0F, 1.0F}},
                                    {{1.0F, 0.0F}, {1.0F}},
                                    0.1F,
                                    1),
                      std::runtime_error);
}

TEST_CASE("feedforward coach learns OR gate", "[feedforward][learning]")
{
    Model network;
    network.addSkill(makeDenseSkill(
        2,
        1,
        {.weights = Pattern::matrix({{0.0F, 0.0F}}), .biases = {0.0F}}));

    const Batch inputs = {{0.0F, 0.0F},
                             {0.0F, 1.0F},
                             {1.0F, 0.0F},
                             {1.0F, 1.0F}};
    const Batch labels = {{0.0F}, {1.0F}, {1.0F}, {1.0F}};

    Coach coach;
    practice(coach, network, inputs, labels, 0.5F, 5000);

    REQUIRE(network.infer({0.0F, 0.0F}).at(0) < 0.5F);
    REQUIRE(network.infer({0.0F, 1.0F}).at(0) > 0.5F);
    REQUIRE(network.infer({1.0F, 0.0F}).at(0) > 0.5F);
    REQUIRE(network.infer({1.0F, 1.0F}).at(0) > 0.5F);
}

TEST_CASE("feedforward coach learns AND gate", "[feedforward][learning]")
{
    Model network;
    network.addSkill(makeDenseSkill(
        2,
        1,
        {.weights = Pattern::matrix({{0.0F, 0.0F}}), .biases = {0.0F}}));

    const Batch inputs = {{0.0F, 0.0F},
                             {0.0F, 1.0F},
                             {1.0F, 0.0F},
                             {1.0F, 1.0F}};
    const Batch labels = {{0.0F}, {0.0F}, {0.0F}, {1.0F}};

    Coach coach;
    practice(coach, network, inputs, labels, 0.5F, 5000);

    REQUIRE(network.infer({0.0F, 0.0F}).at(0) < 0.5F);
    REQUIRE(network.infer({0.0F, 1.0F}).at(0) < 0.5F);
    REQUIRE(network.infer({1.0F, 0.0F}).at(0) < 0.5F);
    REQUIRE(network.infer({1.0F, 1.0F}).at(0) > 0.5F);
}

TEST_CASE("feedforward inference uses trained XOR fixture weights",
          "[feedforward][dense][runtime]")
{
    Model network;
    network.addSkill(makeDenseSkill(
        2,
        2,
        {.weights = Pattern::matrix({{8.051888F, 8.051895F},
                                     {-8.016418F, -8.016412F}}),
         .biases = {-3.967814F, 12.036060F}}));
    network.addSkill(makeDenseSkill(
        2,
        1,
        {.weights = Pattern::matrix({{8.243962F, 8.242302F}}),
         .biases = {-12.212015F}}));

    REQUIRE(network.infer({0.0F, 0.0F}).at(0) < 0.1F);
    REQUIRE(network.infer({0.0F, 1.0F}).at(0) > 0.9F);
    REQUIRE(network.infer({1.0F, 0.0F}).at(0) > 0.9F);
    REQUIRE(network.infer({1.0F, 1.0F}).at(0) < 0.1F);
}
