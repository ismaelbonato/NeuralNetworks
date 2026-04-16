#include "base/ActivationFunction.h"
#include "base/LayerFactory.h"
#include "base/LearningRule.h"
#include "layers/DenseLayer.h"
#include "networks/Perceptron.h"
#include "training/NaturalSelectionTrainer.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>

namespace
{
std::shared_ptr<DenseLayer> makePerceptronLayer()
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
}

TEST_CASE("natural selection trainer selects candidate with lowest squared error",
          "[perceptron][trainer][natural-selection]")
{
    NaturalSelectionTrainer trainer;
    const Batch outputs = {{0.0F, 0.0F, 0.0F, 1.0F},
                           {1.0F, 1.0F, 1.0F, 1.0F},
                           {0.0F, 1.0F, 1.0F, 0.0F},
                           {0.0F, 0.0F, 1.0F, 1.0F}};
    const Batch labels = {{0.0F}, {0.0F}, {0.0F}, {1.0F}};

    REQUIRE(trainer.findClosestPerceptron(outputs, labels) == 0);
}

TEST_CASE("natural selection trainer rejects invalid training data",
          "[perceptron][trainer][natural-selection][errors]")
{
    NaturalSelectionTrainer trainer;
    Perceptron emptyNetwork;

    REQUIRE_THROWS_AS(trainer.learn(emptyNetwork, {{1.0F, 1.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);

    auto layer = makePerceptronLayer();
    Perceptron network(layer);

    REQUIRE_THROWS_AS(trainer.learn(network, {}, {}, 0.1F, 1), std::runtime_error);
    REQUIRE_THROWS_AS(trainer.learn(network, {{1.0F, 1.0F}}, {}, 0.1F, 1),
                      std::runtime_error);
}
