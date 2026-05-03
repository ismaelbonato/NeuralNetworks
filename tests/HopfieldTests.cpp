#include "base/ActivationFunction.h"
#include "base/LayerFactory.h"
#include "layers/HopfieldLayer.h"
#include "base/Model.h"
#include "training/Coach.h"
#include "training/HopfieldPracticePlan.h"
#include "training/ParameterInitializer.h"

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

Skill makeHopfieldSkill(const size_t size)
{
    return makeTrainableSkill<HopfieldLayer>(
        makeHopfieldRecipe(size),
        {.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
         .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>()})
        .intoSkill();
}

Coach makeHopfieldPracticeCoach()
{
    return Coach(std::make_unique<HopfieldPracticePlan>());
}

void practice(Coach &coach,
              Model &network,
              const Batch &patterns,
              Scalar learningRate = Scalar{1.0F},
              size_t epochs = 1)
{
    coach.practice(network,
                   {.inputs = patterns},
                   {.learningRate = learningRate, .epochs = epochs});
}
}

TEST_CASE("hopfield recall updates from current state until convergence", "[hopfield]")
{
    auto skill = makeHopfieldSkill(3);

    REQUIRE(skill.getParameters().biases.empty());
    REQUIRE_THROWS_AS(skill.setParameters({.weights = Pattern::matrix(3, 3, 0.0F),
                                           .biases = {0.0F, 0.0F, 0.0F}}),
                      std::runtime_error);
    skill.setParameters({.weights = Pattern::matrix({{-2.0F, -2.0F, -2.0F},
                                                     {-2.0F, -2.0F, 1.0F},
                                                     {-2.0F, -2.0F, 0.0F}}),
                         .biases = {}});

    Model network;
    network.addSkill(std::move(skill));

    REQUIRE(network.infer({-1.0F, 1.0F, -1.0F})
            == Pattern{-1.0F, 1.0F, 1.0F});
}

TEST_CASE("hopfield rejects patterns with wrong size", "[hopfield][errors]")
{
    Model network;
    network.addSkill(makeHopfieldSkill(4));
    Coach coach = makeHopfieldPracticeCoach();

    REQUIRE_THROWS_AS(practice(coach, network, {{1.0F, -1.0F, 1.0F}}), std::runtime_error);
    REQUIRE_THROWS_AS(network.infer({1.0F, -1.0F, 1.0F}), std::runtime_error);
}

TEST_CASE("hopfield coach keeps diagonal zero and weights symmetric", "[hopfield]")
{
    Model network;
    network.addSkill(makeHopfieldSkill(3));
    Coach coach = makeHopfieldPracticeCoach();

    practice(coach, network, {{1.0F, -1.0F, 1.0F}});

    const Pattern weights = network.getSkill(0).getParameters().weights;
    for (size_t i = 0; i < weights.shape().at(0); ++i) {
        REQUIRE(weights.at({i, i}) == 0.0F);
        for (size_t j = 0; j < weights.shape().at(1); ++j) {
            REQUIRE(weights.at({i, j}) == weights.at({j, i}));
        }
    }
}

TEST_CASE("hopfield coach stores patterns", "[hopfield][coach]")
{
    Model network;
    network.addSkill(makeHopfieldSkill(3));
    Coach coach = makeHopfieldPracticeCoach();

    practice(coach, network, {{1.0F, -1.0F, 1.0F}});

    const Pattern weights = network.getSkill(0).getParameters().weights;
    for (size_t i = 0; i < weights.shape().at(0); ++i) {
        REQUIRE(weights.at({i, i}) == 0.0F);
        for (size_t j = 0; j < weights.shape().at(1); ++j) {
            REQUIRE(weights.at({i, j}) == weights.at({j, i}));
        }
    }
}

TEST_CASE("hopfield coach stores a recalled pattern", "[hopfield]")
{
    Model network;
    network.addSkill(makeHopfieldSkill(4));
    Coach coach = makeHopfieldPracticeCoach();

    const Pattern pattern = {1.0F, -1.0F, 1.0F, -1.0F};

    practice(coach, network, {pattern});

    REQUIRE(network.infer(pattern) == pattern);
}
