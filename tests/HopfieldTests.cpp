#include "base/ActivationFunction.h"
#include "base/LearningRule.h"
#include "base/LayerFactory.h"
#include "layers/HopfieldLayer.h"
#include "base/Model.h"
#include "training/HopfieldTrainer.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace
{
std::unique_ptr<HopfieldLayer> makeHopfieldLayer(const size_t size)
{
    HopfieldLayerConfig config;
    config.name = "test hopfield";
    config.type = "HopfieldLayer";
    config.info = "deterministic test layer";
    config.learningRule = std::make_shared<HebbianRule<Scalar>>();
    config.activation = std::make_shared<StepPolarActivation<Scalar>>();
    config.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    config.biasInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    config.size = size;

    return makeLayer<HopfieldLayer>(config);
}

TrainableLayer &trainableLayer(Model &network)
{
    auto *layer = dynamic_cast<TrainableLayer *>(&network.getLayer(0));
    REQUIRE(layer != nullptr);
    return *layer;
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

    Model network;
    network.addLayer(std::move(layer));

    REQUIRE(network.infer({-1.0F, 1.0F, -1.0F})
            == Pattern{-1.0F, 1.0F, 1.0F});
}

TEST_CASE("hopfield rejects patterns with wrong size", "[hopfield][errors]")
{
    auto layer = makeHopfieldLayer(4);
    Model network;
    network.addLayer(std::move(layer));
    HopfieldTrainer trainer;

    REQUIRE_THROWS_AS(trainer.learn(network, {{1.0F, -1.0F, 1.0F}}), std::runtime_error);
    REQUIRE_THROWS_AS(network.infer({1.0F, -1.0F, 1.0F}), std::runtime_error);
}

TEST_CASE("hopfield trainer keeps diagonal zero and weights symmetric", "[hopfield]")
{
    auto layer = makeHopfieldLayer(3);
    Model network;
    network.addLayer(std::move(layer));
    HopfieldTrainer trainer;

    trainer.learn(network, {{1.0F, -1.0F, 1.0F}});

    const Pattern &weights = trainableLayer(network).getWeights();
    for (size_t i = 0; i < weights.shape().at(0); ++i) {
        REQUIRE(weights.at({i, i}) == 0.0F);
        for (size_t j = 0; j < weights.shape().at(1); ++j) {
            REQUIRE(weights.at({i, j}) == weights.at({j, i}));
        }
    }
}

TEST_CASE("hopfield trainer stores patterns", "[hopfield][trainer]")
{
    auto layer = makeHopfieldLayer(3);
    Model network;
    network.addLayer(std::move(layer));
    HopfieldTrainer trainer;

    trainer.learn(network, {{1.0F, -1.0F, 1.0F}});

    const Pattern &weights = trainableLayer(network).getWeights();
    for (size_t i = 0; i < weights.shape().at(0); ++i) {
        REQUIRE(weights.at({i, i}) == 0.0F);
        for (size_t j = 0; j < weights.shape().at(1); ++j) {
            REQUIRE(weights.at({i, j}) == weights.at({j, i}));
        }
    }
}

TEST_CASE("hopfield trainer stores a recalled pattern", "[hopfield]")
{
    auto layer = makeHopfieldLayer(4);
    Model network;
    network.addLayer(std::move(layer));
    HopfieldTrainer trainer;

    const Pattern pattern = {1.0F, -1.0F, 1.0F, -1.0F};

    trainer.learn(network, {pattern});

    REQUIRE(network.infer(pattern) == pattern);
}
