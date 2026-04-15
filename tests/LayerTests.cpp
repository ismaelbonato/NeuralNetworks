#include "base/ActivationFunction.h"
#include "base/LearningRule.h"
#include "layers/DenseLayer.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>

namespace
{
constexpr Scalar tolerance = 0.0001F;

std::shared_ptr<DenseLayer> makeDenseLayer(const size_t inputSize,
                                           const size_t outputSize,
                                           const bool useBias = true)
{
    LayerConfig config{
        .learningRule = std::make_shared<SGDRule<Scalar>>(),
        .activation = std::make_shared<SigmoidActivation<Scalar>>(),
        .inputSize = inputSize,
        .outputSize = outputSize,
        .name = "test dense layer",
        .type = "DenseLayer",
        .info = "deterministic test layer",
        .useBias = useBias,
        .initWeights = false,
    };

    return std::make_shared<DenseLayer>(config);
}

void requireClose(const Scalar actual, const Scalar expected)
{
    REQUIRE(std::fabs(actual - expected) < tolerance);
}
}

TEST_CASE("dense layer without bias computes weighted sum only", "[layer][dense]")
{
    auto layer = makeDenseLayer(2, 1, false);
    layer->setWeights(Pattern::matrix({{1.0F, 1.0F}}));
    layer->setBiases({10.0F});

    const Pattern output = layer->infer({1.0F, 1.0F});

    requireClose(output[0], 0.880797F);
}

TEST_CASE("dense layer clone preserves weights and biases", "[layer][dense]")
{
    auto layer = makeDenseLayer(2, 2);
    layer->setWeights(Pattern::matrix({{1.0F, 2.0F}, {3.0F, 4.0F}}));
    layer->setBiases({0.5F, -0.5F});

    auto cloned = layer->clone();

    REQUIRE(cloned->getWeights() == layer->getWeights());
    REQUIRE(cloned->getBiases() == layer->getBiases());
}

TEST_CASE("dense layer initializes biases from config", "[layer][dense]")
{
    LayerConfig config{
        .learningRule = std::make_shared<SGDRule<Scalar>>(),
        .activation = std::make_shared<SigmoidActivation<Scalar>>(),
        .inputSize = 2,
        .outputSize = 2,
        .name = "bias init layer",
        .type = "DenseLayer",
        .info = "deterministic test layer",
        .useBias = true,
        .initWeights = false,
        .biasInit = 0.25F,
    };

    DenseLayer layer(config);
    layer.initWeights();

    REQUIRE(layer.getBiases() == Pattern{0.25F, 0.25F});
}

TEST_CASE("layer initializes weights through base implementation", "[layer][dense]")
{
    auto layer = makeDenseLayer(2, 2);

    layer->initWeights();

    REQUIRE(layer->getWeights().hasShape({2, 2}));
    REQUIRE(layer->getBiases().shape() == std::vector<size_t>{2});
}

TEST_CASE("layer initializes weights using configured scale", "[layer][dense]")
{
    LayerConfig config{
        .learningRule = std::make_shared<SGDRule<Scalar>>(),
        .activation = std::make_shared<SigmoidActivation<Scalar>>(),
        .inputSize = 2,
        .outputSize = 2,
        .name = "scaled init layer",
        .type = "DenseLayer",
        .info = "deterministic test layer",
        .useBias = true,
        .initWeights = true,
        .weightInitScale = 0.25F,
    };

    DenseLayer layer(config);
    layer.initWeights();

    for (const Scalar weight : layer.getWeights()) {
        REQUIRE(weight >= -0.25F);
        REQUIRE(weight <= 0.25F);
    }
}

TEST_CASE("layer update rejects mismatched activation and delta sizes", "[layer][errors]")
{
    auto layer = makeDenseLayer(2, 2);
    layer->setWeights(Pattern::matrix({{0.0F, 0.0F}, {0.0F, 0.0F}}));
    layer->setBiases({0.0F, 0.0F});

    REQUIRE_THROWS_AS(layer->updateWeights({1.0F}, {1.0F, 1.0F}, 0.1F),
                      std::runtime_error);
    REQUIRE_THROWS_AS(layer->updateWeights({1.0F, 1.0F}, {1.0F}, 0.1F),
                      std::runtime_error);
}
TEST_CASE("layer setters reject invalid weight and bias shapes", "[layer][errors]")
{
    auto layer = makeDenseLayer(2, 2);

    REQUIRE_THROWS_AS(layer->setWeights(Pattern::vector(4, 0.0F)),
                      std::runtime_error);
    REQUIRE_THROWS_AS(layer->setWeights(Pattern::matrix(1, 4, 0.0F)),
                      std::runtime_error);
    REQUIRE_THROWS_AS(layer->setBiases({0.0F}), std::runtime_error);

    REQUIRE_NOTHROW(layer->setWeights(Pattern::matrix(2, 2, 0.0F)));
    REQUIRE_NOTHROW(layer->setBiases({0.0F, 0.0F}));
}

TEST_CASE("layer reports and rejects missing initialization", "[layer][errors]")
{
    auto layer = makeDenseLayer(2, 1);

    REQUIRE_FALSE(layer->isInitialized());
    REQUIRE_THROWS_AS(layer->requireInitialized(), std::runtime_error);
    REQUIRE_THROWS_AS(layer->infer({1.0F, 1.0F}), std::runtime_error);
    REQUIRE_THROWS_AS(layer->updateWeights({1.0F, 1.0F}, {1.0F}, 0.1F),
                      std::runtime_error);

    layer->initWeights();

    REQUIRE(layer->isInitialized());
    REQUIRE_NOTHROW(layer->requireInitialized());
}
