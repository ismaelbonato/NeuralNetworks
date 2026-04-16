#include "base/ActivationFunction.h"
#include "base/LearningRule.h"
#include "base/LayerFactory.h"
#include "layers/HopfieldLayer.h"
#include "networks/Hopfield.h"
#include "training/HopfieldTrainer.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>

namespace
{
std::shared_ptr<HopfieldLayer> makeHopfieldLayer(const size_t size)
{
    LayerConfig config{
        .learningRule = std::make_shared<HebbianRule<Scalar>>(),
        .activation = std::make_shared<StepPolarActivation<Scalar>>(),
        .inputSize = size,
        .outputSize = size,
        .name = "test hopfield",
        .type = "HopfieldLayer",
        .info = "deterministic test layer",
        .weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
        .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
    };

    return makeLayer<HopfieldLayer>(config);
}
}

TEST_CASE("hopfield recall updates from current state until convergence", "[hopfield]")
{
    auto layer = makeHopfieldLayer(3);

    REQUIRE(layer->getBiases().empty());
    REQUIRE_THROWS_AS(layer->setBiases({0.0F, 0.0F, 0.0F}), std::runtime_error);
    layer->setWeights(Pattern::matrix({{-2.0F, -2.0F, -2.0F},
                                      {-2.0F, -2.0F, 1.0F},
                                      {-2.0F, -2.0F, 0.0F}}));

    Hopfield network(layer);

    REQUIRE(network.infer({-1.0F, 1.0F, -1.0F})
            == Pattern{-1.0F, 1.0F, 1.0F});
}

TEST_CASE("hopfield rejects patterns with wrong size", "[hopfield][errors]")
{
    auto layer = makeHopfieldLayer(4);
    Hopfield network(layer);
    HopfieldTrainer trainer;

    REQUIRE_THROWS_AS(trainer.learn(network, {{1.0F, -1.0F, 1.0F}}), std::runtime_error);
    REQUIRE_THROWS_AS(network.infer({1.0F, -1.0F, 1.0F}), std::runtime_error);
}

TEST_CASE("hopfield trainer keeps diagonal zero and weights symmetric", "[hopfield]")
{
    auto layer = makeHopfieldLayer(3);
    Hopfield network(layer);
    HopfieldTrainer trainer;

    trainer.learn(network, {{1.0F, -1.0F, 1.0F}});

    for (size_t i = 0; i < layer->getWeights().shape()[0]; ++i) {
        REQUIRE(layer->getWeights().at({i, i}) == 0.0F);
        for (size_t j = 0; j < layer->getWeights().shape()[1]; ++j) {
            REQUIRE(layer->getWeights().at({i, j}) == layer->getWeights().at({j, i}));
        }
    }
}

TEST_CASE("hopfield trainer stores patterns", "[hopfield][trainer]")
{
    auto layer = makeHopfieldLayer(3);
    Hopfield network(layer);
    HopfieldTrainer trainer;

    trainer.learn(network, {{1.0F, -1.0F, 1.0F}});

    for (size_t i = 0; i < layer->getWeights().shape()[0]; ++i) {
        REQUIRE(layer->getWeights().at({i, i}) == 0.0F);
        for (size_t j = 0; j < layer->getWeights().shape()[1]; ++j) {
            REQUIRE(layer->getWeights().at({i, j}) == layer->getWeights().at({j, i}));
        }
    }
}

TEST_CASE("hopfield trainer stores a recalled pattern", "[hopfield]")
{
    auto layer = makeHopfieldLayer(4);
    Hopfield network(layer);
    HopfieldTrainer trainer;

    const Pattern pattern = {1.0F, -1.0F, 1.0F, -1.0F};

    trainer.learn(network, {pattern});

    REQUIRE(network.infer(pattern) == pattern);
}
