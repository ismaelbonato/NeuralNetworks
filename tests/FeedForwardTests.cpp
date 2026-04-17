#include "base/ActivationFunction.h"
#include "base/LearningRule.h"
#include "base/LayerFactory.h"
#include "layers/DenseLayer.h"
#include "base/Model.h"
#include "training/FeedforwardTrainer.h"

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
    auto rule = std::make_shared<SGDRule<Scalar>>();
    auto activation = std::make_shared<SigmoidActivation<Scalar>>();
    LayerConfig config{
        .learningRule = rule,
        .activation = activation,
        .inputSize = inputSize,
        .outputSize = outputSize,
        .name = "test dense layer",
        .type = "DenseLayer",
        .info = "deterministic test layer",
    };

    if (!randomInitialize) {
        config.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>();
        config.biasInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    }

    auto layer = makeLayer<DenseLayer>(config);
    return layer;
}

void requireClose(const Scalar actual, const Scalar expected)
{
    REQUIRE(std::fabs(actual - expected) < tolerance);
}
}

TEST_CASE("dense layer computes deterministic weighted sums and activations",
          "[feedforward][dense]")
{
    auto layer = makeDenseLayer(2, 2);
    layer->setWeights(Pattern::matrix({{1.0F, -1.0F}, {0.5F, 0.5F}}));
    layer->setBiases({0.0F, -0.5F});

    const Pattern output = layer->infer({2.0F, 1.0F});

    requireClose(output.at(0), 0.7310586F);
    requireClose(output.at(1), 0.7310586F);
}

TEST_CASE("feedforward inference composes dense layers", "[feedforward]")
{
    auto hidden = makeDenseLayer(2, 1);
    hidden->setWeights(Pattern::matrix({{1.0F, 1.0F}}));
    hidden->setBiases({0.0F});

    auto output = makeDenseLayer(1, 1);
    output->setWeights(Pattern::matrix({{2.0F}}));
    output->setBiases({-1.0F});

    Model network;
    network.addLayer(std::move(hidden));
    network.addLayer(std::move(output));

    const Pattern prediction = network.infer({1.0F, 1.0F});

    requireClose(prediction.at(0), 0.6816998F);
}

TEST_CASE("feedforward trainer updates single layer through SGD",
          "[feedforward][learning]")
{
    auto layer = makeDenseLayer(1, 1);
    Model network;
    network.addLayer(std::move(layer));
    FeedforwardTrainer trainer;

    trainer.learn(network, {{1.0F}}, {{1.0F}}, 1.0F, 1);

    requireClose(network.getLayer(0).getWeights().at({0, 0}), 0.125F);
    requireClose(network.getLayer(0).getBiases().at(0), 0.125F);
}

TEST_CASE("feedforward inference rejects missing layers and invalid input sizes",
          "[feedforward][errors]")
{
    Model emptyNetwork;

    REQUIRE_THROWS_AS(emptyNetwork.infer({1.0F}), std::runtime_error);

    auto layer = makeDenseLayer(2, 1);
    Model network;
    network.addLayer(std::move(layer));

    REQUIRE_THROWS_AS(network.infer({1.0F}), std::runtime_error);
}

TEST_CASE("feedforward trainer can train the same model more than once",
          "[feedforward][learning]")
{
    auto layer = makeDenseLayer(1, 1);
    Model network;
    network.addLayer(std::move(layer));
    FeedforwardTrainer trainer;

    trainer.learn(network, {{1.0F}}, {{1.0F}}, 1.0F, 1);
    trainer.learn(network, {{1.0F}}, {{1.0F}}, 1.0F, 1);

    REQUIRE(network.getLayer(0).getWeights().at({0, 0}) != 0.0F);
}

TEST_CASE("feedforward trainer direct API updates weights and biases",
          "[feedforward][trainer]")
{
    auto layer = makeDenseLayer(1, 1);
    Model network;
    network.addLayer(std::move(layer));
    FeedforwardTrainer trainer;

    trainer.learn(network, {{1.0F}}, {{1.0F}}, 1.0F, 1);

    requireClose(network.getLayer(0).getWeights().at({0, 0}), 0.125F);
    requireClose(network.getLayer(0).getBiases().at(0), 0.125F);
}

TEST_CASE("feedforward trainer updates hidden and output layers",
          "[feedforward][learning]")
{
    auto hidden = makeDenseLayer(2, 2);
    hidden->setWeights(Pattern::matrix({{0.1F, -0.2F}, {0.3F, 0.4F}}));
    hidden->setBiases({0.0F, 0.0F});

    auto output = makeDenseLayer(2, 1);
    output->setWeights(Pattern::matrix({{0.5F, -0.3F}}));
    output->setBiases({0.0F});

    const Scalar hiddenWeightBefore = hidden->getWeights().at({0, 0});
    const Scalar outputWeightBefore = output->getWeights().at({0, 0});

    Model network;
    network.addLayer(std::move(hidden));
    network.addLayer(std::move(output));
    FeedforwardTrainer trainer;
    trainer.learn(network, {{1.0F, 0.0F}}, {{1.0F}}, 0.5F, 1);

    REQUIRE(network.getLayer(0).getWeights().at({0, 0}) != hiddenWeightBefore);
    REQUIRE(network.getLayer(1).getWeights().at({0, 0}) != outputWeightBefore);
}

TEST_CASE("feedforward trainer rejects invalid training data", "[feedforward][errors]")
{
    auto layer = makeDenseLayer(1, 1);
    Model network;
    network.addLayer(std::move(layer));
    FeedforwardTrainer trainer;

    REQUIRE_THROWS_AS(trainer.learn(network, {}, {}, 0.1F, 1), std::runtime_error);
    REQUIRE_THROWS_AS(trainer.learn(network, {{1.0F}}, {}, 0.1F, 1), std::runtime_error);

    Model emptyNetwork;
    REQUIRE_THROWS_AS(trainer.learn(emptyNetwork, {{1.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);
}

TEST_CASE("feedforward trainer rejects wrong input and label shapes", "[feedforward][errors]")
{
    auto layer = makeDenseLayer(2, 2);
    Model network;
    network.addLayer(std::move(layer));
    FeedforwardTrainer trainer;

    REQUIRE_THROWS_AS(trainer.learn(network, {{1.0F}}, {{1.0F, 0.0F}}, 0.1F, 1),
                      std::runtime_error);
    REQUIRE_THROWS_AS(trainer.learn(network, {{1.0F, 0.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);
    REQUIRE_THROWS_AS(trainer.learn(network, {{1.0F, 0.0F}, {1.0F}},
                                    {{1.0F, 0.0F}, {0.0F, 1.0F}},
                                    0.1F,
                                    1),
                      std::runtime_error);
    REQUIRE_THROWS_AS(trainer.learn(network, {{1.0F, 0.0F}, {0.0F, 1.0F}},
                                    {{1.0F, 0.0F}, {1.0F}},
                                    0.1F,
                                    1),
                      std::runtime_error);
}

TEST_CASE("feedforward trainer learns OR gate", "[feedforward][learning]")
{
    auto layer = makeDenseLayer(2, 1);
    layer->setWeights(Pattern::matrix({{0.0F, 0.0F}}));
    layer->setBiases({0.0F});

    Model network;
    network.addLayer(std::move(layer));

    const Batch inputs = {{0.0F, 0.0F},
                             {0.0F, 1.0F},
                             {1.0F, 0.0F},
                             {1.0F, 1.0F}};
    const Batch labels = {{0.0F}, {1.0F}, {1.0F}, {1.0F}};

    FeedforwardTrainer trainer;
    trainer.learn(network, inputs, labels, 0.5F, 5000);

    REQUIRE(network.infer({0.0F, 0.0F}).at(0) < 0.5F);
    REQUIRE(network.infer({0.0F, 1.0F}).at(0) > 0.5F);
    REQUIRE(network.infer({1.0F, 0.0F}).at(0) > 0.5F);
    REQUIRE(network.infer({1.0F, 1.0F}).at(0) > 0.5F);
}

TEST_CASE("feedforward trainer learns AND gate", "[feedforward][learning]")
{
    auto layer = makeDenseLayer(2, 1);
    layer->setWeights(Pattern::matrix({{0.0F, 0.0F}}));
    layer->setBiases({0.0F});

    Model network;
    network.addLayer(std::move(layer));

    const Batch inputs = {{0.0F, 0.0F},
                             {0.0F, 1.0F},
                             {1.0F, 0.0F},
                             {1.0F, 1.0F}};
    const Batch labels = {{0.0F}, {0.0F}, {0.0F}, {1.0F}};

    FeedforwardTrainer trainer;
    trainer.learn(network, inputs, labels, 0.5F, 5000);

    REQUIRE(network.infer({0.0F, 0.0F}).at(0) < 0.5F);
    REQUIRE(network.infer({0.0F, 1.0F}).at(0) < 0.5F);
    REQUIRE(network.infer({1.0F, 0.0F}).at(0) < 0.5F);
    REQUIRE(network.infer({1.0F, 1.0F}).at(0) > 0.5F);
}
