#include "base/ActivationFunction.h"
#include "base/LayerFactory.h"
#include "base/Model.h"
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
    DenseLayerRecipe denseRecipe;
    denseRecipe.name = "test dense layer";
    denseRecipe.type = "DenseLayer";
    denseRecipe.info = "deterministic test layer";
    denseRecipe.activation = std::make_shared<SigmoidActivation<Scalar>>();
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

TEST_CASE("dense layer adds recipe bias to pre-activation", "[layer][dense]")
{
    auto layer = makeDenseLayer(2, 1);
    const Parameters parameters{
        .weights = Pattern::matrix({{1.0F, 1.0F}}),
        .biases = {10.0F},
    };

    const Pattern output = layer->infer({1.0F, 1.0F}, parameters);

    requireClose(output.at(0), 0.9999938F);
}

TEST_CASE("skill parameter snapshots preserve weights and biases", "[skill][dense]")
{
    Skill source(makeDenseLayer(2, 2));
    source.setParameters({
        .weights = Pattern::matrix({{1.0F, 2.0F}, {3.0F, 4.0F}}),
        .biases = {0.5F, -0.5F},
    });

    Skill target(makeDenseLayer(2, 2));
    target.setParameters(source.getParameters());

    REQUIRE(target.getParameters().weights == source.getParameters().weights);
    REQUIRE(target.getParameters().biases == source.getParameters().biases);
}

TEST_CASE("layer recipe derives flat sizes from explicit shapes", "[layer][shape]")
{
    DenseLayerRecipe config;
    config.name = "shape recipe dense layer";
    config.type = "DenseLayer";
    config.info = "shape-only test layer";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.expectedInputShape = {2};
    config.expectedOutputShape = {1};

    auto layer = makeLayer<DenseLayer>(config);

    REQUIRE(layer->getInputSize() == 2);
    REQUIRE(layer->getOutputSize() == 1);
    REQUIRE(layer->getInputShape().dimensions == std::vector<size_t>{2});
    REQUIRE(layer->getOutputShape().dimensions == std::vector<size_t>{1});
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

TEST_CASE("skill parameter setter rejects invalid weight and bias shapes",
          "[skill][errors]")
{
    Skill skill(makeDenseLayer(2, 2));

    REQUIRE_THROWS_AS(skill.setParameters(
                          {.weights = Pattern::vector(4, 0.0F),
                           .biases = {0.0F, 0.0F}}),
                      std::runtime_error);
    REQUIRE_THROWS_AS(skill.setParameters(
                          {.weights = Pattern::matrix(1, 4, 0.0F),
                           .biases = {0.0F, 0.0F}}),
                      std::runtime_error);
    REQUIRE_THROWS_AS(skill.setParameters(
                          {.weights = Pattern::matrix(2, 2, 0.0F),
                           .biases = {0.0F}}),
                      std::runtime_error);

    REQUIRE_NOTHROW(skill.setParameters(
        {.weights = Pattern::matrix(2, 2, 0.0F),
         .biases = {0.0F, 0.0F}}));
}

TEST_CASE("skill performs through its runtime layer", "[skill][runtime]")
{
    auto layer = makeDenseLayer(2, 1);
    Skill skill(std::move(layer));
    skill.setParameters({
        .weights = Pattern::matrix({{1.0F, 1.0F}}),
        .biases = {0.0F},
    });

    const Pattern output = skill.perform({1.0F, 1.0F});

    requireClose(output.at(0), 0.880797F);
}

TEST_CASE("skill exposes parameter snapshots for parameterized layers",
          "[skill][parameters]")
{
    auto layer = makeDenseLayer(2, 1);
    Skill skill(std::move(layer));
    const Parameters parameters{
        .weights = Pattern::matrix({{1.0F, -1.0F}}),
        .biases = {0.5F},
    };

    skill.setParameters(parameters);

    REQUIRE(skill.hasParameters());
    REQUIRE(skill.parameters().has_value());
    REQUIRE(skill.getParameters().weights == parameters.weights);
    REQUIRE(skill.getParameters().biases == parameters.biases);
    REQUIRE_NOTHROW(skill.requireInitialized());
}

TEST_CASE("skill-owned parameters drive runtime execution",
          "[skill][parameters]")
{
    auto layer = makeDenseLayer(1, 1);
    Skill skill(std::move(layer));
    skill.setParameters({
        .weights = Pattern::matrix({{2.0F}}),
        .biases = {-1.0F},
    });

    const Pattern output = skill.perform({1.0F});

    requireClose(output.at(0), 0.7310586F);
    REQUIRE(skill.getParameters().weights == Pattern::matrix({{2.0F}}));
    REQUIRE(skill.getParameters().biases == Pattern{-1.0F});
}

TEST_CASE("parameterized runtime layer still requires external parameters",
          "[skill][parameters]")
{
    auto layer = makeDenseLayer(1, 1);
    Skill skill(std::move(layer));
    skill.setParameters({
        .weights = Pattern::matrix({{2.0F}}),
        .biases = {-1.0F},
    });
    const auto &wrappedLayer = dynamic_cast<const DenseLayer &>(skill.layer());

    const Pattern output = skill.perform({1.0F});

    requireClose(output.at(0), 0.7310586F);
    REQUIRE_THROWS_AS(wrappedLayer.infer({1.0F}), std::runtime_error);
}

TEST_CASE("runtime-only skills do not expose parameters",
          "[skill][parameters]")
{
    auto layer = makeFlattenLayer({1, 2});
    Skill skill(std::move(layer));

    REQUIRE_FALSE(skill.hasParameters());
    REQUIRE_FALSE(skill.parameters().has_value());
    REQUIRE_NOTHROW(skill.requireInitialized());
    REQUIRE_THROWS_AS(skill.getParameters(), std::runtime_error);
    REQUIRE_THROWS_AS(skill.setParameters({}), std::runtime_error);
}

TEST_CASE("model can infer through added skills", "[model][skill]")
{
    DenseLayerRecipe config;
    config.name = "skill dense layer";
    config.type = "DenseLayer";
    config.info = "skill construction test";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputSize = 1;
    config.outputSize = 1;

    Skill runtimeSkill(makeLayer<DenseLayer>(config));
    runtimeSkill.setParameters({
        .weights = Pattern::matrix({{2.0F}}),
        .biases = {-1.0F},
    });

    Model network;
    network.addSkill(std::move(runtimeSkill));

    const Pattern output = network.infer({1.0F});

    REQUIRE(network.numLayers() == 1);
    REQUIRE(&network.getSkill(0).layer() == &network.getLayer(0));
    requireClose(output.at(0), 0.7310586F);
}

TEST_CASE("layer guard rejects derived layers that skip initialization", "[layer][errors]")
{
    DenseLayerRecipe config;
    config.name = "uninitialized test layer";
    config.type = "TestLayer";
    config.info = "intentionally skips construction initialization";
    config.activation = std::make_shared<SigmoidActivation<Scalar>>();
    config.inputSize = 2;
    config.outputSize = 1;
    auto layer = std::make_unique<UninitializedLayer>(config);
    Skill skill(std::move(layer));

    REQUIRE_THROWS_AS(skill.requireInitialized(), std::runtime_error);
    REQUIRE_THROWS_AS(skill.perform({1.0F, 1.0F}), std::runtime_error);
    Model network;
    network.addSkill(std::move(skill));
}
