#include "base/ActivationFunction.h"
#include "base/LearningRule.h"
#include "layers/DenseLayer.h"
#include "networks/Perceptron.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>

namespace
{
std::shared_ptr<DenseLayer> makePerceptronLayer(const size_t outputSize = 1)
{
    LayerConfig config{
        .learningRule = std::make_shared<PerceptronRule<Scalar>>(),
        .activation = std::make_shared<StepActivation<Scalar>>(),
        .inputSize = 2,
        .outputSize = outputSize,
        .name = "test perceptron",
        .type = "DenseLayer",
        .info = "deterministic test layer",
        .useBias = true,
        .initWeights = false,
    };

    auto layer = std::make_shared<DenseLayer>(config);
    layer->initWeights();
    return layer;
}
}

TEST_CASE("perceptron learns AND gate", "[perceptron]")
{
    auto layer = makePerceptronLayer();
    layer->weights = {{0.0F, 0.0F}};
    layer->biases = {0.0F};

    Perceptron network(layer);
    const Patterns inputs = {{0.0F, 0.0F}, {0.0F, 1.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}};
    const Patterns labels = {{0.0F}, {0.0F}, {0.0F}, {1.0F}};

    network.learn(inputs, labels, 0.1F, 20);

    REQUIRE(network.infer({0.0F, 0.0F})[0] == 0.0F);
    REQUIRE(network.infer({0.0F, 1.0F})[0] == 0.0F);
    REQUIRE(network.infer({1.0F, 0.0F})[0] == 0.0F);
    REQUIRE(network.infer({1.0F, 1.0F})[0] == 1.0F);
}

TEST_CASE("perceptron rejects multi-output layers", "[perceptron][errors]")
{
    auto layer = makePerceptronLayer(2);
    Perceptron network(layer);

    REQUIRE_THROWS_AS(network.learn({{1.0F, 1.0F}}, {{1.0F, 0.0F}}, 0.1F, 1),
                      std::runtime_error);
}

TEST_CASE("perceptron learn rejects invalid training data", "[perceptron][errors]")
{
    auto layer = makePerceptronLayer();
    Perceptron network(layer);

    REQUIRE_THROWS_AS(network.learn({}, {}, 0.1F, 1), std::runtime_error);
    REQUIRE_THROWS_AS(network.learn({{1.0F, 1.0F}}, {}, 0.1F, 1),
                      std::runtime_error);

    Perceptron emptyNetwork;
    REQUIRE_THROWS_AS(emptyNetwork.learn({{1.0F, 1.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);
}
