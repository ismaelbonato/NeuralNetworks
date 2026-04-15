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

    auto layer = std::make_shared<DenseLayer>(config);
    layer->initWeights();
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

    requireClose(output[0], 0.7310586F);
    requireClose(output[1], 0.7310586F);
}

TEST_CASE("feedforward inference composes dense layers", "[feedforward]")
{
    auto hidden = makeDenseLayer(2, 1);
    hidden->setWeights(Pattern::matrix({{1.0F, 1.0F}}));
    hidden->setBiases({0.0F});

    auto output = makeDenseLayer(1, 1);
    output->setWeights(Pattern::matrix({{2.0F}}));
    output->setBiases({-1.0F});

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

    requireClose(layer->getWeights().at({0, 0}), 0.125F);
    requireClose(layer->getBiases()[0], 0.125F);
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

class InspectableFeedforward : public Feedforward
{
public:
    using Feedforward::Feedforward;

    size_t activationBufferSize() const { return activate.size(); }
    size_t preActivationBufferSize() const { return preActivations.size(); }
};

TEST_CASE("feedforward can learn more than once without growing training buffers",
          "[feedforward][learning]")
{
    auto layer = makeDenseLayer(1, 1);
    InspectableFeedforward network({layer});

    network.learn({{1.0F}}, {{1.0F}}, 1.0F, 1);
    network.learn({{1.0F}}, {{1.0F}}, 1.0F, 1);

    REQUIRE(network.activationBufferSize() == 2);
    REQUIRE(network.preActivationBufferSize() == 1);
}

TEST_CASE("feedforward learning updates hidden and output layers",
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

    Feedforward network({hidden, output});
    network.learn({{1.0F, 0.0F}}, {{1.0F}}, 0.5F, 1);

    REQUIRE(hidden->getWeights().at({0, 0}) != hiddenWeightBefore);
    REQUIRE(output->getWeights().at({0, 0}) != outputWeightBefore);
}

TEST_CASE("feedforward learn rejects invalid training data", "[feedforward][errors]")
{
    auto layer = makeDenseLayer(1, 1);
    Feedforward network({layer});

    REQUIRE_THROWS_AS(network.learn({}, {}, 0.1F, 1), std::runtime_error);
    REQUIRE_THROWS_AS(network.learn({{1.0F}}, {}, 0.1F, 1), std::runtime_error);

    Feedforward emptyNetwork;
    REQUIRE_THROWS_AS(emptyNetwork.learn({{1.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);
}

TEST_CASE("feedforward learn rejects wrong input and label shapes", "[feedforward][errors]")
{
    auto layer = makeDenseLayer(2, 2);
    Feedforward network({layer});

    REQUIRE_THROWS_AS(network.learn({{1.0F}}, {{1.0F, 0.0F}}, 0.1F, 1),
                      std::runtime_error);
    REQUIRE_THROWS_AS(network.learn({{1.0F, 0.0F}}, {{1.0F}}, 0.1F, 1),
                      std::runtime_error);
    REQUIRE_THROWS_AS(network.learn({{1.0F, 0.0F}, {1.0F}},
                                    {{1.0F, 0.0F}, {0.0F, 1.0F}},
                                    0.1F,
                                    1),
                      std::runtime_error);
    REQUIRE_THROWS_AS(network.learn({{1.0F, 0.0F}, {0.0F, 1.0F}},
                                    {{1.0F, 0.0F}, {1.0F}},
                                    0.1F,
                                    1),
                      std::runtime_error);
}

TEST_CASE("feedforward learns OR gate", "[feedforward][learning]")
{
    auto layer = makeDenseLayer(2, 1);
    layer->setWeights(Pattern::matrix({{0.0F, 0.0F}}));
    layer->setBiases({0.0F});

    Feedforward network({layer});

    const Batch inputs = {{0.0F, 0.0F},
                             {0.0F, 1.0F},
                             {1.0F, 0.0F},
                             {1.0F, 1.0F}};
    const Batch labels = {{0.0F}, {1.0F}, {1.0F}, {1.0F}};

    network.learn(inputs, labels, 0.5F, 5000);

    REQUIRE(network.infer({0.0F, 0.0F})[0] < 0.5F);
    REQUIRE(network.infer({0.0F, 1.0F})[0] > 0.5F);
    REQUIRE(network.infer({1.0F, 0.0F})[0] > 0.5F);
    REQUIRE(network.infer({1.0F, 1.0F})[0] > 0.5F);
}

TEST_CASE("feedforward learns AND gate", "[feedforward][learning]")
{
    auto layer = makeDenseLayer(2, 1);
    layer->setWeights(Pattern::matrix({{0.0F, 0.0F}}));
    layer->setBiases({0.0F});

    Feedforward network({layer});

    const Batch inputs = {{0.0F, 0.0F},
                             {0.0F, 1.0F},
                             {1.0F, 0.0F},
                             {1.0F, 1.0F}};
    const Batch labels = {{0.0F}, {0.0F}, {0.0F}, {1.0F}};

    network.learn(inputs, labels, 0.5F, 5000);

    REQUIRE(network.infer({0.0F, 0.0F})[0] < 0.5F);
    REQUIRE(network.infer({0.0F, 1.0F})[0] < 0.5F);
    REQUIRE(network.infer({1.0F, 0.0F})[0] < 0.5F);
    REQUIRE(network.infer({1.0F, 1.0F})[0] > 0.5F);
}
