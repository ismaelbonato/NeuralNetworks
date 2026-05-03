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

Coach makeNaturalSelectionPracticeCoach(NaturalSelectionConfig config = {})
{
    return Coach(std::make_unique<NaturalSelectionPracticePlan>(config));
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

TEST_CASE("natural selection coach rejects invalid training data",
          "[perceptron][coach][natural-selection][errors]")
{
    Coach coach = makeNaturalSelectionPracticeCoach();
    Model emptyNetwork;

    REQUIRE_THROWS_AS(practice(coach, emptyNetwork, {{1.0F, 1.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);

    Model network;
    network.addSkill(makePerceptronSkill());

    REQUIRE_THROWS_AS(practice(coach, network, {}, {}, 0.1F, 1), std::runtime_error);
    REQUIRE_THROWS_AS(practice(coach, network, {{1.0F, 1.0F}}, {}, 0.1F, 1),
                      std::runtime_error);
    REQUIRE_THROWS_AS(practice(coach, network, {{1.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);
    REQUIRE_THROWS_AS(practice(coach, network, {{1.0F, 1.0F}}, {{1.0F, 0.0F}}, 0.1F, 1),
                      std::runtime_error);
}

TEST_CASE("natural selection coach rejects invalid configuration",
          "[coach][natural-selection][errors]")
{
    Model network;
    network.addSkill(makePerceptronSkill());
    Coach coach = makeNaturalSelectionPracticeCoach({.populationSize = 0});

    REQUIRE_THROWS_AS(practice(coach, network, {{1.0F, 1.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);
}

TEST_CASE("natural selection coach supports multi-output models",
          "[coach][natural-selection]")
{
    Model network;
    network.addSkill(makeMultiOutputSkill());
    Coach coach = makeNaturalSelectionPracticeCoach({.populationSize = 2});

    REQUIRE_NOTHROW(practice(coach,
                                  network,
                                  {{1.0F, 0.0F}},
                                  {{1.0F, 0.0F}},
                                  0.0F,
                                  1));
}
