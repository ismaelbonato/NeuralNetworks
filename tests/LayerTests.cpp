#include "base/ActivationFunction.h"
#include "base/LearningRule.h"
#include "base/LayerFactory.h"
#include "layers/DenseLayer.h"
#include "layers/FlattenLayer.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>

namespace
{
constexpr Scalar tolerance = 0.0001F;

std::unique_ptr<DenseLayer> makeDenseLayer(const size_t inputSize,
                                           const size_t outputSize)
{
    DenseLayerConfig denseConfig;
    denseConfig.name = "test dense layer";
    denseConfig.type = "DenseLayer";
    denseConfig.info = "deterministic test layer";
    denseConfig.learningRule = std::make_shared<SGDRule<Scalar>>();
    denseConfig.activation = std::make_shared<SigmoidActivation<Scalar>>();
    denseConfig.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    denseConfig.biasInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    denseConfig.inputSize = inputSize;
    denseConfig.outputSize = outputSize;

    return makeLayer<DenseLayer>(denseConfig);
}

std::unique_ptr<FlattenLayer> makeFlattenLayer(const Shape &inputShape)
{
    FlattenLayerConfig config;
    config.name = "test flatten layer";
    config.type = "FlattenLayer";
    config.info = "deterministic test layer";
    config.expectedInputShape = inputShape;

    return makeLayer<FlattenLayer>(config);
}

void requireClose(const Scalar actual, const Scalar expected)
{
    REQUIRE(std::fabs(actual - expected) < tolerance);
}

class UninitializedLayer : public TrainableLayer
{
public:
    explicit UninitializedLayer(const DenseLayerConfig &newConfig)
        : TrainableLayer(newConfig, {newConfig.inputSize}, {newConfig.outputSize})
    {}

protected:
    Shape expectedWeightShape() const override
    {
        return {getOutputSize(), getInputSize()};
    }

    Shape expectedBiasShape() const override
    {
        return {getOutputSize()};
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
    DenseLayerConfig config;
    config.name = "bias init layer";
    config.type = "DenseLayer";
    config.info = "deterministic test layer";
    config.learningRule = std::make_shared<SGDRule<Scalar>>();
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    config.biasInitializer = std::make_shared<ConstantInitializer<Scalar>>(0.25F);
    config.inputSize = 2;
    config.outputSize = 2;

    auto layer = makeLayer<DenseLayer>(config);

    REQUIRE(layer->getBiases() == Pattern{0.25F, 0.25F});
}

TEST_CASE("layer initializes weights through base implementation", "[layer][dense]")
{
    auto layer = makeDenseLayer(2, 2);


    REQUIRE(layer->getWeights().hasShape({2, 2}));
    REQUIRE(layer->getBiases().shape() == std::vector<size_t>{2});
}

TEST_CASE("layer config derives flat sizes from explicit shapes", "[layer][shape]")
{
    DenseLayerConfig config;
    config.name = "shape configured dense layer";
    config.type = "DenseLayer";
    config.info = "shape-only test layer";
    config.learningRule = std::make_shared<SGDRule<Scalar>>();
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    config.biasInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    config.expectedInputShape = {2};
    config.expectedOutputShape = {1};

    auto layer = makeLayer<DenseLayer>(config);

    REQUIRE(layer->getInputSize() == 2);
    REQUIRE(layer->getOutputSize() == 1);
    REQUIRE(layer->getInputShape().dimensions == std::vector<size_t>{2});
    REQUIRE(layer->getOutputShape().dimensions == std::vector<size_t>{1});
    REQUIRE(layer->getWeights().hasShape({1, 2}));
}

TEST_CASE("layer config rejects inconsistent flat size and shape", "[layer][shape][errors]")
{
    DenseLayerConfig config;
    config.name = "invalid shape configured dense layer";
    config.type = "DenseLayer";
    config.info = "shape mismatch test layer";
    config.learningRule = std::make_shared<SGDRule<Scalar>>();
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputSize = 3;
    config.outputSize = 1;
    config.expectedInputShape = {2};
    config.expectedOutputShape = {1};

    REQUIRE_THROWS_AS(makeLayer<DenseLayer>(config), std::invalid_argument);
}

TEST_CASE("flatten layer reshapes explicit input shape to a vector", "[layer][flatten]")
{
    auto layer = makeFlattenLayer({2, 2});
    auto input = Pattern::withShape({2, 2});
    input.at({0, 0}) = 1.0F;
    input.at({0, 1}) = 2.0F;
    input.at({1, 0}) = 3.0F;
    input.at({1, 1}) = 4.0F;

    const Pattern output = layer->infer(input);

    REQUIRE_FALSE(layer->isTrainable());
    REQUIRE(output.shape() == std::vector<size_t>{4});
    REQUIRE(output == Pattern{1.0F, 2.0F, 3.0F, 4.0F});
}

TEST_CASE("flatten layer restores previous activation shape during backward pass",
          "[layer][flatten]")
{
    auto layer = makeFlattenLayer({2, 2});
    const Pattern delta = {1.0F, 2.0F, 3.0F, 4.0F};
    const auto previousActivation = Pattern::withShape({2, 2});

    const Pattern previousDelta = layer->backwardPass(delta, previousActivation);

    REQUIRE(previousDelta.shape() == std::vector<size_t>{2, 2});
    REQUIRE(previousDelta.at({0, 0}) == 1.0F);
    REQUIRE(previousDelta.at({1, 1}) == 4.0F);
}

TEST_CASE("layer initializes weights using configured scale", "[layer][dense]")
{
    DenseLayerConfig config;
    config.name = "scaled init layer";
    config.type = "DenseLayer";
    config.info = "deterministic test layer";
    config.learningRule = std::make_shared<SGDRule<Scalar>>();
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.weightInitializer = std::make_shared<UniformInitializer<Scalar>>(-0.25F, 0.25F);
    config.inputSize = 2;
    config.outputSize = 2;

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
    DenseLayerConfig config;
    config.name = "uninitialized test layer";
    config.type = "TestLayer";
    config.info = "intentionally skips construction initialization";
    config.learningRule = std::make_shared<SGDRule<Scalar>>();
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    config.biasInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    config.inputSize = 2;
    config.outputSize = 1;
    auto layer = std::make_shared<UninitializedLayer>(config);

    REQUIRE_FALSE(layer->isInitialized());
    REQUIRE_THROWS_AS(layer->requireInitialized(), std::runtime_error);
    REQUIRE_THROWS_AS(layer->infer({1.0F, 1.0F}), std::runtime_error);
    REQUIRE_THROWS_AS(layer->updateWeights({1.0F, 1.0F}, {1.0F}, 0.1F),
                      std::runtime_error);
}
