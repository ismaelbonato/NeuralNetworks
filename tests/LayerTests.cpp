#include "base/ActivationFunction.h"
#include "base/LearningRule.h"
#include "base/LayerFactory.h"
#include "layers/DenseLayer.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>

namespace
{
constexpr Scalar tolerance = 0.0001F;

std::unique_ptr<DenseLayer> makeDenseLayer(const size_t inputSize,
                                           const size_t outputSize)
{
    LayerConfig config{
        .learningRule = std::make_shared<SGDRule<Scalar>>(),
        .activation = std::make_shared<SigmoidActivation<Scalar>>(),
        .inputSize = inputSize,
        .outputSize = outputSize,
        .name = "test dense layer",
        .type = "DenseLayer",
        .info = "deterministic test layer",
        .weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
        .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
    };

    return makeLayer<DenseLayer>(config);
}

void requireClose(const Scalar actual, const Scalar expected)
{
    REQUIRE(std::fabs(actual - expected) < tolerance);
}

class UninitializedLayer : public Layer
{
public:
    explicit UninitializedLayer(const LayerConfig &newConfig)
        : Layer(newConfig)
    {}

protected:
    Shape expectedWeightShape() const override
    {
        return {config.outputSize, config.inputSize};
    }

    Shape expectedBiasShape() const override
    {
        return {config.outputSize};
    }
};
}

TEST_CASE("dense layer adds configured bias to weighted sum", "[layer][dense]")
{
    auto layer = makeDenseLayer(2, 1);
    layer->setWeights(Pattern::matrix({{1.0F, 1.0F}}));
    layer->setBiases({10.0F});

    const Pattern output = layer->infer({1.0F, 1.0F});

    requireClose(output.at(0), 0.9999938F);
}

TEST_CASE("layer parameter snapshots preserve weights and biases", "[layer][dense]")
{
    auto source = makeDenseLayer(2, 2);
    source->setWeights(Pattern::matrix({{1.0F, 2.0F}, {3.0F, 4.0F}}));
    source->setBiases({0.5F, -0.5F});

    auto target = makeDenseLayer(2, 2);
    target->setParameters(source->getParameters());

    REQUIRE(target->getWeights() == source->getWeights());
    REQUIRE(target->getBiases() == source->getBiases());
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
        .weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
        .biasInitializer = std::make_shared<ConstantInitializer<Scalar>>(0.25F),
    };

    auto layer = makeLayer<DenseLayer>(config);

    REQUIRE(layer->getBiases() == Pattern{0.25F, 0.25F});
}

TEST_CASE("layer initializes weights through base implementation", "[layer][dense]")
{
    auto layer = makeDenseLayer(2, 2);


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
        .weightInitializer = std::make_shared<UniformInitializer<Scalar>>(-0.25F, 0.25F),
    };

    auto layer = makeLayer<DenseLayer>(config);

    for (const Scalar weight : layer->getWeights()) {
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

TEST_CASE("factory initializes dense layer", "[layer][dense]")
{
    auto layer = makeDenseLayer(2, 1);

    REQUIRE(layer->isInitialized());
    REQUIRE_NOTHROW(layer->requireInitialized());
}

TEST_CASE("layer guard rejects derived layers that skip initialization", "[layer][errors]")
{
    LayerConfig config{
        .learningRule = std::make_shared<SGDRule<Scalar>>(),
        .activation = std::make_shared<SigmoidActivation<Scalar>>(),
        .inputSize = 2,
        .outputSize = 1,
        .name = "uninitialized test layer",
        .type = "TestLayer",
        .info = "intentionally skips construction initialization",
        .weightInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
        .biasInitializer = std::make_shared<ZeroInitializer<Scalar>>(),
    };
    auto layer = std::make_shared<UninitializedLayer>(config);

    REQUIRE_FALSE(layer->isInitialized());
    REQUIRE_THROWS_AS(layer->requireInitialized(), std::runtime_error);
    REQUIRE_THROWS_AS(layer->infer({1.0F, 1.0F}), std::runtime_error);
    REQUIRE_THROWS_AS(layer->updateWeights({1.0F, 1.0F}, {1.0F}, 0.1F),
                      std::runtime_error);
}
