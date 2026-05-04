#include "base/ActivationFunction.h"
#include "base/LayerFactory.h"
#include "layers/DenseLayer.h"
#include "base/Model.h"
#include "training/ParameterInitializer.h"

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

TEST_CASE("feedforward inference rejects missing layers and invalid input sizes",
          "[feedforward][errors]")
{
    Model emptyNetwork;

    REQUIRE_THROWS_AS(emptyNetwork.infer({1.0F}), std::runtime_error);

    Model network;
    network.addSkill(makeDenseSkill(2, 1));

    REQUIRE_THROWS_AS(network.infer({1.0F}), std::runtime_error);
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
