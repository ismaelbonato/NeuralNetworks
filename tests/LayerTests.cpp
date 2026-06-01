#include "base/ActivationFunction.h"
#include "base/Model.h"
#include "layers/DenseLayer.h"
#include "layers/FlattenLayer.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>

using namespace nn;

namespace {
constexpr Scalar tolerance = 0.0001F;

std::unique_ptr<DenseLayer> makeDenseLayer(const size_t inputSize,
                                           const size_t outputSize)
{
    DenseLayerRecipe denseRecipe;
    denseRecipe.name = "test dense layer";
    denseRecipe.type = "DenseLayer";
    denseRecipe.info = "deterministic test layer";
    denseRecipe.activation = std::make_shared<SigmoidActivation<Scalar>>();
    denseRecipe.inputShape = {inputSize};
    denseRecipe.outputShape = {outputSize};

    return std::make_unique<DenseLayer>(denseRecipe);
}

std::unique_ptr<FlattenLayer> makeFlattenLayer(const Shape &inputShape)
{
    FlattenLayerRecipe config;
    config.name = "test flatten layer";
    config.type = "FlattenLayer";
    config.info = "deterministic test layer";
    config.inputShape = inputShape;

    return std::make_unique<FlattenLayer>(config);
}

void requireClose(const Scalar actual, const Scalar expected)
{
    REQUIRE(std::fabs(actual - expected) < tolerance);
}

class MissingParametersLayer : public DenseLayer
{
public:
    explicit MissingParametersLayer(const DenseLayerRecipe &newRecipe)
        : DenseLayer(newRecipe)
    {}
};

struct DelegatingLayerRecipe : LayerRecipe
{
    DelegatingLayerRecipe()
    {
        name = "test delegating layer";
        type = "DelegatingLayer";
        info = "test layer for base inference delegation";
    }

    Shape getInputShape() const override { return {2}; }
    Shape getOutputShape() const override { return {2}; }
    void validateRecipe() const override
    {
        LayerRecipe::validateRecipe();
    }
};

class DelegatingLayer : public Layer
{
public:
    DelegatingLayer()
        : Layer(std::make_unique<DelegatingLayerRecipe>())
    {}

    size_t forwardCalls() const { return calls; }

protected:
    Pattern forward(const Pattern &input) const override
    {
        ++calls;
        return input + Pattern{1.0F, 2.0F};
    }

private:
    mutable size_t calls = 0;
};
} // namespace

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

TEST_CASE("dense layer adds recipe bias to pre-activation", "[layer][dense]")
{
    auto layer = makeDenseLayer(2, 1);
    layer->setParameters({
        .weights = Pattern::matrix({{1.0F, 1.0F}}),
        .biases = {10.0F},
    });

    const Pattern output = layer->infer({1.0F, 1.0F});

    requireClose(output.at(0), 0.9999938F);
}

TEST_CASE("layer parameter snapshots preserve weights and biases",
          "[layer][dense]")
{
    auto source = makeDenseLayer(2, 2);
    source->setParameters({
        .weights = Pattern::matrix({{1.0F, 2.0F}, {3.0F, 4.0F}}),
        .biases = {0.5F, -0.5F},
    });

    auto target = makeDenseLayer(2, 2);
    target->setParameters(source->getParameters());

    REQUIRE(target->getParameters().weights == source->getParameters().weights);
    REQUIRE(target->getParameters().biases == source->getParameters().biases);
}

TEST_CASE("layer recipe derives flat sizes from explicit shapes",
          "[layer][shape]")
{
    DenseLayerRecipe config;
    config.name = "shape recipe dense layer";
    config.type = "DenseLayer";
    config.info = "shape-only test layer";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputShape = {2};
    config.outputShape = {1};

    auto layer = std::make_unique<DenseLayer>(config);

    REQUIRE(layer->getInputShape().elementCount() == 2);
    REQUIRE(layer->getOutputShape().elementCount() == 1);
    REQUIRE(layer->getInputShape().dimensions == std::vector<size_t>{2});
    REQUIRE(layer->getOutputShape().dimensions == std::vector<size_t>{1});
}

TEST_CASE("layer recipe rejects invalid dense shapes",
          "[layer][shape][errors]")
{
    DenseLayerRecipe config;
    config.name = "invalid shape recipe dense layer";
    config.type = "DenseLayer";
    config.info = "invalid shape test layer";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputShape = {0};
    config.outputShape = {1};

    REQUIRE_THROWS_AS(std::make_unique<DenseLayer>(config),
                      std::invalid_argument);
}

TEST_CASE("flatten layer reshapes explicit input shape to a vector",
          "[layer][flatten]")
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

TEST_CASE("layer parameter setter rejects invalid weight and bias shapes",
          "[layer][errors]")
{
    auto layer = makeDenseLayer(2, 2);

    REQUIRE_THROWS_AS(layer->setParameters({.weights = Pattern::vector(4, 0.0F),
                                            .biases = {0.0F, 0.0F}}),
                      std::runtime_error);
    REQUIRE_THROWS_AS(layer->setParameters(
                          {.weights = Pattern::matrix(1, 4, 0.0F),
                           .biases = {0.0F, 0.0F}}),
                      std::runtime_error);
    REQUIRE_THROWS_AS(layer->setParameters(
                          {.weights = Pattern::matrix(2, 2, 0.0F),
                           .biases = {0.0F}}),
                      std::runtime_error);

    REQUIRE_NOTHROW(layer->setParameters(
        {.weights = Pattern::matrix(2, 2, 0.0F), .biases = {0.0F, 0.0F}}));
}

TEST_CASE("parameterized layer infers through owned parameters",
          "[layer][runtime]")
{
    auto layer = makeDenseLayer(2, 1);
    layer->setParameters({
        .weights = Pattern::matrix({{1.0F, 1.0F}}),
        .biases = {0.0F},
    });

    const Pattern output = layer->infer({1.0F, 1.0F});

    requireClose(output.at(0), 0.880797F);
}

TEST_CASE("layer exposes parameter snapshots for parameterized layers",
          "[layer][parameters]")
{
    auto layer = makeDenseLayer(2, 1);
    const Parameters parameters{
        .weights = Pattern::matrix({{1.0F, -1.0F}}),
        .biases = {0.5F},
    };

    layer->setParameters(parameters);

    REQUIRE(layer->usesParameters());
    REQUIRE(layer->parameters().has_value());
    REQUIRE(layer->getParameters().weights == parameters.weights);
    REQUIRE(layer->getParameters().biases == parameters.biases);
    REQUIRE_NOTHROW(layer->requireParameters());
}

TEST_CASE("layer-owned parameters drive runtime execution",
          "[layer][parameters]")
{
    auto layer = makeDenseLayer(1, 1);
    layer->setParameters({
        .weights = Pattern::matrix({{2.0F}}),
        .biases = {-1.0F},
    });

    const Pattern output = layer->infer({1.0F});

    requireClose(output.at(0), 0.7310586F);
    REQUIRE(layer->getParameters().weights == Pattern::matrix({{2.0F}}));
    REQUIRE(layer->getParameters().biases == Pattern{-1.0F});
}

TEST_CASE("parameterized runtime layer uses reassigned owned parameters",
          "[layer][parameters]")
{
    auto layer = makeDenseLayer(1, 1);
    layer->setParameters({
        .weights = Pattern::matrix({{2.0F}}),
        .biases = {-1.0F},
    });

    const Pattern output = layer->infer({1.0F});

    layer->setParameters({
        .weights = Pattern::matrix({{1.0F}}),
        .biases = {0.0F},
    });
    const Pattern reassignedOutput = layer->infer({1.0F});

    requireClose(output.at(0), 0.7310586F);
    requireClose(reassignedOutput.at(0), 0.7310586F);
}

TEST_CASE("runtime-only layers do not expose parameters", "[layer][parameters]")
{
    auto layer = makeFlattenLayer({1, 2});

    REQUIRE_FALSE(layer->usesParameters());
    REQUIRE_FALSE(layer->parameters().has_value());
    REQUIRE_NOTHROW(layer->requireParameters());
    REQUIRE_THROWS_AS(layer->getParameters(), std::runtime_error);
    REQUIRE_THROWS_AS(layer->setParameters({}), std::runtime_error);
}

TEST_CASE("model can infer through added layers", "[model][layer]")
{
    DenseLayerRecipe config;
    config.name = "model dense layer";
    config.type = "DenseLayer";
    config.info = "model construction test";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputShape = {1};
    config.outputShape = {1};

    auto layer = std::make_unique<DenseLayer>(config);
    layer->setParameters({
        .weights = Pattern::matrix({{2.0F}}),
        .biases = {-1.0F},
    });

    Model network;
    network.addLayer(std::move(layer));

    const Pattern output = network.infer({1.0F});

    REQUIRE(network.numLayers() == 1);
    requireClose(output.at(0), 0.7310586F);
}

TEST_CASE("layer guard rejects parameter layers without assigned weights",
          "[layer][errors]")
{
    DenseLayerRecipe config;
    config.name = "missing parameters test layer";
    config.type = "TestLayer";
    config.info = "intentionally skips parameter assignment";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputShape = {2};
    config.outputShape = {1};
    auto layer = std::make_unique<MissingParametersLayer>(config);

    REQUIRE_THROWS_AS(layer->requireParameters(), std::runtime_error);
    REQUIRE_THROWS_AS(layer->infer({1.0F, 1.0F}), std::runtime_error);
    Model network;
    network.addLayer(std::move(layer));
}
