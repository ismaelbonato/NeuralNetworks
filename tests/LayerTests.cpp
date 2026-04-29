#include "base/ActivationFunction.h"
#include "base/LearningRule.h"
#include "base/LayerFactory.h"
#include "base/Model.h"
#include "layers/DenseLayer.h"
#include "layers/FlattenLayer.h"
#include "training/GradientEngine.h"
#include "training/Optimizer.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>

namespace
{
constexpr Scalar tolerance = 0.0001F;

std::unique_ptr<DenseLayer> makeDenseLayer(const size_t inputSize,
                                           const size_t outputSize)
{
    DenseLayerRecipe denseRecipe;
    denseRecipe.name = "test dense layer";
    denseRecipe.type = "DenseLayer";
    denseRecipe.info = "deterministic test layer";
    denseRecipe.activation = std::make_shared<SigmoidActivation<Scalar>>();
    denseRecipe.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    denseRecipe.biasInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    denseRecipe.inputSize = inputSize;
    denseRecipe.outputSize = outputSize;

    return makeLayer<DenseLayer>(denseRecipe);
}

std::unique_ptr<FlattenLayer> makeFlattenLayer(const Shape &inputShape)
{
    FlattenLayerRecipe config;
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

LearningRuleOptimizer makeSgdOptimizer()
{
    return LearningRuleOptimizer{std::make_shared<SGDRule<Scalar>>()};
}

class UninitializedLayer : public DenseLayer
{
public:
    explicit UninitializedLayer(const DenseLayerRecipe &newRecipe)
        : DenseLayer(newRecipe)
    {}
};

class DelegatingLayer : public Layer
{
public:
    DelegatingLayer()
        : Layer(LayerRecipe{}, {2}, {2})
    {}

    size_t forwardCalls() const
    {
        return calls;
    }

protected:
    Pattern forward(const Pattern &input) const override
    {
        ++calls;
        return input + Pattern{1.0F, 2.0F};
    }

private:
    mutable size_t calls = 0;
};
}

TEST_CASE("base layer infer rejects shape mismatches before delegation",
          "[layer][infer][errors]")
{
    DelegatingLayer layer;

    REQUIRE_THROWS_AS(layer.infer({1.0F}), std::runtime_error);
    REQUIRE(layer.forwardCalls() == 0);
}

TEST_CASE("base layer infer delegates valid input to forward", "[layer][infer]")
{
    DelegatingLayer layer;

    const Pattern output = layer.infer({3.0F, 5.0F});

    REQUIRE(layer.forwardCalls() == 1);
    REQUIRE(output == Pattern{4.0F, 7.0F});
}

TEST_CASE("gradient engine rejects unsupported layer backpropagation",
          "[gradient][errors]")
{
    DelegatingLayer layer;
    const BackpropagationGradientEngine gradientEngine;

    REQUIRE_THROWS_AS(gradientEngine.backwardThroughLayer(layer,
                                                          Pattern{1.0F, 1.0F},
                                                          Pattern{1.0F, 1.0F}),
                      std::runtime_error);
}

TEST_CASE("dense layer adds recipe bias to pre-activation", "[layer][dense]")
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

TEST_CASE("dense layer initializes biases from recipe", "[layer][dense]")
{
    DenseLayerRecipe config;
    config.name = "bias init layer";
    config.type = "DenseLayer";
    config.info = "deterministic test layer";
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

TEST_CASE("layer recipe derives flat sizes from explicit shapes", "[layer][shape]")
{
    DenseLayerRecipe config;
    config.name = "shape recipe dense layer";
    config.type = "DenseLayer";
    config.info = "shape-only test layer";
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

TEST_CASE("layer recipe rejects inconsistent flat size and shape", "[layer][shape][errors]")
{
    DenseLayerRecipe config;
    config.name = "invalid shape recipe dense layer";
    config.type = "DenseLayer";
    config.info = "shape mismatch test layer";
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

    REQUIRE(output.shape() == std::vector<size_t>{4});
    REQUIRE(output == Pattern{1.0F, 2.0F, 3.0F, 4.0F});
}

TEST_CASE("flatten layer restores previous activation shape during backward pass",
          "[layer][flatten]")
{
    auto layer = makeFlattenLayer({2, 2});
    const Pattern delta = {1.0F, 2.0F, 3.0F, 4.0F};
    const auto previousActivation = Pattern::withShape({2, 2});

    const BackpropagationGradientEngine gradientEngine;
    const Pattern previousDelta = gradientEngine.backwardThroughLayer(*layer,
                                                                      delta,
                                                                      previousActivation);

    REQUIRE(previousDelta.shape() == std::vector<size_t>{2, 2});
    REQUIRE(previousDelta.at({0, 0}) == 1.0F);
    REQUIRE(previousDelta.at({1, 1}) == 4.0F);
}

TEST_CASE("layer initializes weights using recipe scale", "[layer][dense]")
{
    DenseLayerRecipe config;
    config.name = "scaled init layer";
    config.type = "DenseLayer";
    config.info = "deterministic test layer";
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

TEST_CASE("optimizer step rejects mismatched activation and delta sizes",
          "[optimizer][errors]")
{
    auto layer = makeDenseLayer(2, 2);
    layer->setWeights(Pattern::matrix({{0.0F, 0.0F}, {0.0F, 0.0F}}));
    layer->setBiases({0.0F, 0.0F});
    Model network;
    network.addLayer(std::move(layer));
    const auto optimizer = makeSgdOptimizer();

    REQUIRE_THROWS_AS(optimizer.step(network,
                                     {{1.0F}},
                                     {{1.0F, 1.0F}},
                                     0.1F),
                      std::runtime_error);
    REQUIRE_THROWS_AS(optimizer.step(network,
                                     {{1.0F, 1.0F}},
                                     {{1.0F}},
                                     0.1F),
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
    DenseLayerRecipe config;
    config.name = "uninitialized test layer";
    config.type = "TestLayer";
    config.info = "intentionally skips construction initialization";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.weightInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    config.biasInitializer = std::make_shared<ZeroInitializer<Scalar>>();
    config.inputSize = 2;
    config.outputSize = 1;
    auto layer = std::make_unique<UninitializedLayer>(config);

    REQUIRE_FALSE(layer->isInitialized());
    REQUIRE_THROWS_AS(layer->requireInitialized(), std::runtime_error);
    REQUIRE_THROWS_AS(layer->infer({1.0F, 1.0F}), std::runtime_error);
    Model network;
    network.addLayer(std::move(layer));
    const auto optimizer = makeSgdOptimizer();
    REQUIRE_THROWS_AS(optimizer.step(network,
                                     {{1.0F, 1.0F}},
                                     {{1.0F}},
                                     0.1F),
                      std::runtime_error);
}
