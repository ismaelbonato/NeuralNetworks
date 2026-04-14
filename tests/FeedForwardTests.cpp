#include "base/ActivationFunction.h"
#include "base/LearningRule.h"
#include "layers/DenseLayer.h"
#include "networks/FeedForward.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <stdexcept>

namespace
{
constexpr Scalar tolerance = 0.0001F;

std::shared_ptr<DenseLayer> makeDenseLayer(const size_t inputSize,
                                           const size_t outputSize,
                                           const bool initWeights = false)
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
        .useBias = true,
        .initWeights = initWeights,
    };

    return std::make_shared<DenseLayer>(config);
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
    layer->weights = {{1.0F, -1.0F}, {0.5F, 0.5F}};
    layer->biases = {0.0F, -0.5F};

    const Pattern output = layer->infer({2.0F, 1.0F});

    requireClose(output[0], 0.7310586F);
    requireClose(output[1], 0.7310586F);
}

TEST_CASE("feedforward inference composes dense layers", "[feedforward]")
{
    auto hidden = makeDenseLayer(2, 1);
    hidden->weights = {{1.0F, 1.0F}};
    hidden->biases = {0.0F};

    auto output = makeDenseLayer(1, 1);
    output->weights = {{2.0F}};
    output->biases = {-1.0F};

    Feedforward network({hidden, output});

    const Pattern prediction = network.infer({1.0F, 1.0F});

    requireClose(prediction[0], 0.6816998F);
}

TEST_CASE("feedforward learning updates weights and biases through SGD",
          "[feedforward][learning]")
{
    auto layer = makeDenseLayer(1, 1);
    Feedforward network({layer});

    network.learn({{1.0F}}, {{1.0F}}, 1.0F, 1);

    requireClose(layer->weights[0][0], 0.125F);
    requireClose(layer->biases[0], 0.125F);
}

TEST_CASE("feedforward inference rejects missing layers and invalid input sizes",
          "[feedforward][errors]")
{
    Feedforward emptyNetwork;

    REQUIRE_THROWS_AS(emptyNetwork.infer({1.0F}), std::runtime_error);

    auto layer = makeDenseLayer(2, 1);
    Feedforward network({layer});

    REQUIRE_THROWS_AS(network.infer({1.0F}), std::runtime_error);
}
