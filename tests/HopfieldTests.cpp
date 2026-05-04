#include "base/ActivationFunction.h"
#include "layers/HopfieldLayer.h"
#include "base/Model.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace
{
HopfieldLayerRecipe makeHopfieldRecipe(const size_t size)
{
    HopfieldLayerRecipe config;
    config.name = "test hopfield";
    config.type = "HopfieldLayer";
    config.info = "deterministic test layer";
    config.activation = std::make_shared<StepPolarActivation<Scalar>>();
    config.size = size;

    return config;
}

std::unique_ptr<HopfieldLayer> makeHopfieldLayer(const size_t size)
{
    return std::make_unique<HopfieldLayer>(makeHopfieldRecipe(size));
}

}

TEST_CASE("hopfield recall updates from current state until convergence", "[hopfield]")
{
    auto layer = makeHopfieldLayer(3);

    REQUIRE(layer->getParameters().biases.empty());
    REQUIRE_THROWS_AS(layer->setParameters({.weights = Pattern::matrix(3, 3, 0.0F),
                                           .biases = {0.0F, 0.0F, 0.0F}}),
                      std::runtime_error);
    layer->setParameters({.weights = Pattern::matrix({{-2.0F, -2.0F, -2.0F},
                                                     {-2.0F, -2.0F, 1.0F},
                                                     {-2.0F, -2.0F, 0.0F}}),
                         .biases = {}});

    Model network;
    network.addLayer(std::move(layer));

    REQUIRE(network.infer({-1.0F, 1.0F, -1.0F})
            == Pattern{-1.0F, 1.0F, 1.0F});
}

TEST_CASE("hopfield rejects patterns with wrong size", "[hopfield][errors]")
{
    Model network;
    network.addLayer(makeHopfieldLayer(4));

    REQUIRE_THROWS_AS(network.infer({1.0F, -1.0F, 1.0F}), std::runtime_error);
}

TEST_CASE("hopfield inference uses static stored 3-value pattern weights",
          "[hopfield][runtime]")
{
    auto layer = makeHopfieldLayer(3);
    layer->setParameters({.weights = Pattern::matrix({{0.0F, -1.0F, 1.0F},
                                                     {-1.0F, 0.0F, -1.0F},
                                                     {1.0F, -1.0F, 0.0F}}),
                         .biases = {}});

    Model network;
    network.addLayer(std::move(layer));

    REQUIRE(network.infer({1.0F, -1.0F, 1.0F})
            == Pattern{1.0F, -1.0F, 1.0F});
}

TEST_CASE("hopfield inference uses static stored 4-value pattern weights",
          "[hopfield][runtime]")
{
    auto layer = makeHopfieldLayer(4);
    layer->setParameters({.weights = Pattern::matrix({{0.0F, -1.0F, 1.0F, -1.0F},
                                                     {-1.0F, 0.0F, -1.0F, 1.0F},
                                                     {1.0F, -1.0F, 0.0F, -1.0F},
                                                     {-1.0F, 1.0F, -1.0F, 0.0F}}),
                         .biases = {}});

    Model network;
    network.addLayer(std::move(layer));

    REQUIRE(network.infer({1.0F, -1.0F, 1.0F, -1.0F})
            == Pattern{1.0F, -1.0F, 1.0F, -1.0F});
}
